#include "sih/bnb/branch_and_bound.hpp"
#include "sih/bnb/milp_presolve.hpp"
#include "sih/bnb/gomory_cuts.hpp"
#include "sih/bnb/heuristics.hpp"
#include "sih/bnb/thread_pool.hpp"
#include "sih/simplex/simplex_solver.hpp"
#include <chrono>
#include <cmath>
#include <queue>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <iostream>

namespace sih {
namespace bnb {

namespace {

struct CompareBestFirst {
    bool operator()(const BnBNode& a, const BnBNode& b) const noexcept {
        return a.lp_bound > b.lp_bound; // Min-heap: lowest bound first (for minimization)
    }
};

bool check_integrality(const std::vector<double>& x,
                       const std::vector<model::VariableType>& vt,
                       double tol) {
    for (size_t j = 0; j < vt.size(); ++j) {
        if (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary) {
            double v = x[j];
            if (std::abs(v - std::round(v)) > tol) {
                return false;
            }
        }
    }
    return true;
}

int64_t select_most_fractional(const std::vector<double>& x,
                               const std::vector<model::VariableType>& vt,
                               double tol) {
    int64_t best_j = -1;
    double max_dist_from_int = 0.0;

    for (size_t j = 0; j < vt.size(); ++j) {
        if (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary) {
            double v = x[j];
            double frac = v - std::floor(v);
            double dist = std::abs(frac - 0.5);
            double int_dist = std::abs(v - std::round(v));
            if (int_dist > tol) {
                // Closer to 0.5 means dist is smaller
                if (best_j == -1 || dist < max_dist_from_int) {
                    max_dist_from_int = dist;
                    best_j = static_cast<int64_t>(j);
                }
            }
        }
    }
    return best_j;
}

} // namespace

model::Solution BranchAndBound::solve(const model::Problem& problem,
                                     const model::Options& options) {
    auto start_time = std::chrono::high_resolution_clock::now();
    model::Options opts = options;

    model::Solution final_sol;
    final_sol.status = model::SolutionStatus::Unknown;

    // 1. MILP Presolve
    auto presolve_res = MilpPresolver::presolve(problem, opts.zero_tol);
    if (presolve_res.is_infeasible) {
        final_sol.status = model::SolutionStatus::Infeasible;
        return final_sol;
    }

    model::Problem curr_prob = std::move(presolve_res.problem);
    int64_t n = curr_prob.num_cols();
    const auto& vt = curr_prob.var_types();

    // 2. Solve Root LP Relaxation
    auto root_sol = simplex::SimplexSolver::solve(curr_prob, options);
    if (!root_sol.is_optimal()) {
        final_sol.status = root_sol.status;
        return final_sol;
    }

    int64_t total_iterations = root_sol.simplex_iterations;

    // Check if naturally integer feasible
    if (check_integrality(root_sol.x, vt, opts.integrality_tol)) {
        root_sol.nodes_explored = 1;
        root_sol.dual_bound = root_sol.primal_objective;
        root_sol.mip_gap = 0.0;
        return root_sol;
    }

    // 3. Apply Root Gomory Cuts
    if (opts.strategy.cut_rounds > 0) {
        root_sol = GomoryCutGenerator::apply_root_cuts(curr_prob, root_sol, options);
        total_iterations += root_sol.simplex_iterations;

        if (check_integrality(root_sol.x, vt, opts.integrality_tol)) {
            root_sol.nodes_explored = 1;
            root_sol.dual_bound = root_sol.primal_objective;
            root_sol.mip_gap = 0.0;
            return root_sol;
        }
    }

    // 4. Seed Incumbent via Heuristics
    bool has_incumbent = false;
    double best_obj = model::SIH_INFINITY;
    std::vector<double> best_x(n, 0.0);

    auto h_round = PrimalHeuristics::simple_rounding(curr_prob, root_sol, opts.integrality_tol);
    if (h_round.found_incumbent && h_round.objective < best_obj) {
        has_incumbent = true;
        best_obj = h_round.objective;
        best_x = std::move(h_round.solution);
    }

    auto h_dive = PrimalHeuristics::fractional_diving(curr_prob, root_sol, options, 30);
    if (h_dive.found_incumbent && h_dive.objective < best_obj) {
        has_incumbent = true;
        best_obj = h_dive.objective;
        best_x = std::move(h_dive.solution);
    }

    // 5. Initialize Branch-and-Bound Tree
    std::mutex tree_mutex;
    std::priority_queue<BnBNode, std::vector<BnBNode>, CompareBestFirst> best_first_queue;
    std::vector<BnBNode> depth_first_stack;

    BnBNode root_node;
    root_node.id = 0;
    root_node.depth = 0;
    root_node.lp_bound = root_sol.primal_objective;
    root_node.col_lower = curr_prob.col_lower();
    root_node.col_upper = curr_prob.col_upper();
    root_node.col_basis = root_sol.col_basis;
    root_node.row_basis = root_sol.row_basis;

    if (opts.strategy.node_selection == model::NodeSelection::BestBound) {
        best_first_queue.push(std::move(root_node));
    } else {
        depth_first_stack.push_back(std::move(root_node));
    }

    std::atomic<int64_t> node_counter{1};
    std::atomic<int64_t> nodes_explored{0};
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> time_limit_hit{false};
    std::atomic<bool> node_limit_hit{false};

    double global_dual_bound = root_sol.primal_objective;

    // Pseudocost state
    std::vector<double> pc_down(n, 1.0);
    std::vector<double> pc_up(n, 1.0);
    std::vector<int64_t> pc_down_cnt(n, 0);
    std::vector<int64_t> pc_up_cnt(n, 0);

    // Node exploration worker function
    auto explore_node = [&](BnBNode node) {
        nodes_explored++;

        // Pruning by bound
        {
            std::lock_guard<std::mutex> lock(tree_mutex);
            if (has_incumbent && node.lp_bound >= best_obj - 1e-6) {
                return;
            }
        }

        // Subproblem formulation with child bounds
        model::Problem sub_prob = curr_prob;
        sub_prob.col_lower() = node.col_lower;
        sub_prob.col_upper() = node.col_upper;

        // Solve child LP relaxation using Dual Simplex warm-start from parent basis
        auto child_sol = simplex::SimplexSolver::solve_from_basis(sub_prob, node.col_basis, node.row_basis, options);
        total_iterations += child_sol.simplex_iterations;

        if (!child_sol.is_optimal()) {
            return; // Pruned by infeasibility
        }

        double child_obj = child_sol.primal_objective;

        // Phase 8: Backpropagate pseudocost from branching decision
        if (node.branch_var >= 0 && node.parent_obj < 1e30) {
            double delta_obj = child_obj - node.parent_obj;
            if (delta_obj > 0.0) {
                std::lock_guard<std::mutex> lock(tree_mutex);
                if (node.branch_dir < 0) {
                    // Down branch
                    double frac = node.branch_frac_down;
                    if (frac > 1e-8) {
                        double unit_gain = delta_obj / frac;
                        pc_down[node.branch_var] = (pc_down[node.branch_var] * pc_down_cnt[node.branch_var] + unit_gain) /
                                                   (pc_down_cnt[node.branch_var] + 1);
                        pc_down_cnt[node.branch_var]++;
                    }
                } else {
                    // Up branch
                    double frac = node.branch_frac_up;
                    if (frac > 1e-8) {
                        double unit_gain = delta_obj / frac;
                        pc_up[node.branch_var] = (pc_up[node.branch_var] * pc_up_cnt[node.branch_var] + unit_gain) /
                                                 (pc_up_cnt[node.branch_var] + 1);
                        pc_up_cnt[node.branch_var]++;
                    }
                }
            }
        }


        // Check if bound exceeds incumbent
        {
            std::lock_guard<std::mutex> lock(tree_mutex);
            if (has_incumbent && child_obj >= best_obj - 1e-6) {
                return; // Pruned by bound
            }
        }

        // Check integrality
        if (check_integrality(child_sol.x, vt, opts.integrality_tol)) {
            // Found integer feasible solution!
            std::lock_guard<std::mutex> lock(tree_mutex);
            if (child_obj < best_obj) {
                has_incumbent = true;
                best_obj = child_obj;
                best_x = child_sol.x;
            }
            return; // Pruned by integrality
        }

        // Select branching variable using Reliability Branching
        // (Phase 8 improvement: strong branching to initialize pseudocosts)
        int64_t branch_var = -1;
        const int64_t rel_count = 8;       // reliability threshold
        const int64_t max_sb_cands = 10;   // max strong branching candidates per node
        const int64_t max_sb_iters = 100;  // iteration limit for strong branching LP

        if (opts.strategy.branching_rule == model::BranchingRule::PseudoCost ||
            opts.strategy.branching_rule == model::BranchingRule::Reliability) {
            double best_score = -1.0;

            // Collect fractional integer candidates
            struct BranchCandidate {
                int64_t var;
                double frac_down;
                double frac_up;
                double pc_score;
                bool needs_sb;
            };
            std::vector<BranchCandidate> candidates;
            candidates.reserve(n);

            for (int64_t j = 0; j < n; ++j) {
                if (vt[j] == model::VariableType::Integer || vt[j] == model::VariableType::Binary) {
                    double v = child_sol.x[j];
                    double f_down = v - std::floor(v);
                    double f_up = 1.0 - f_down;
                    if (f_down > opts.integrality_tol && f_up > opts.integrality_tol) {
                        double d_down = pc_down[j] * f_down;
                        double d_up = pc_up[j] * f_up;
                        double score = (5.0 / 6.0) * std::min(d_down, d_up) + (1.0 / 6.0) * std::max(d_down, d_up);
                        bool needs = (pc_down_cnt[j] < rel_count || pc_up_cnt[j] < rel_count);
                        candidates.push_back({j, f_down, f_up, score, needs});
                    }
                }
            }

            // Sort candidates: unreliable first (need strong branching), then by score
            std::sort(candidates.begin(), candidates.end(), [](const BranchCandidate& a, const BranchCandidate& b) {
                if (a.needs_sb != b.needs_sb) return a.needs_sb > b.needs_sb;
                return a.pc_score > b.pc_score;
            });

            // Strong branching for unreliable candidates (limited to max_sb_cands)
            int64_t sb_done = 0;
            for (auto& cand : candidates) {
                double score = cand.pc_score;

                if (cand.needs_sb && sb_done < max_sb_cands && node.depth < 20) {
                    // Perform strong branching: solve both child LPs with limited iterations
                    model::Options sb_opts = options;
                    sb_opts.iteration_limit = max_sb_iters;
                    sb_opts.log_to_console = false;

                    double floor_v = std::floor(child_sol.x[cand.var]);
                    double ceil_v = std::ceil(child_sol.x[cand.var]);

                    // Down branch: x_j <= floor
                    model::Problem sb_down_prob = sub_prob;
                    sb_down_prob.col_upper()[cand.var] = floor_v;
                    auto sb_down_sol = simplex::SimplexSolver::solve_from_basis(
                        sb_down_prob, child_sol.col_basis, child_sol.row_basis, sb_opts);
                    total_iterations += sb_down_sol.simplex_iterations;

                    // Up branch: x_j >= ceil
                    model::Problem sb_up_prob = sub_prob;
                    sb_up_prob.col_lower()[cand.var] = ceil_v;
                    auto sb_up_sol = simplex::SimplexSolver::solve_from_basis(
                        sb_up_prob, child_sol.col_basis, child_sol.row_basis, sb_opts);
                    total_iterations += sb_up_sol.simplex_iterations;

                    // Update pseudocosts from strong branching results
                    if (sb_down_sol.is_optimal() && cand.frac_down > 1e-8) {
                        double delta_down = (sb_down_sol.primal_objective - child_obj) / cand.frac_down;
                        if (delta_down > 0.0) {
                            std::lock_guard<std::mutex> lock(tree_mutex);
                            pc_down[cand.var] = (pc_down[cand.var] * pc_down_cnt[cand.var] + delta_down) /
                                                (pc_down_cnt[cand.var] + 1);
                            pc_down_cnt[cand.var]++;
                        }
                    }
                    if (sb_up_sol.is_optimal() && cand.frac_up > 1e-8) {
                        double delta_up = (sb_up_sol.primal_objective - child_obj) / cand.frac_up;
                        if (delta_up > 0.0) {
                            std::lock_guard<std::mutex> lock(tree_mutex);
                            pc_up[cand.var] = (pc_up[cand.var] * pc_up_cnt[cand.var] + delta_up) /
                                              (pc_up_cnt[cand.var] + 1);
                            pc_up_cnt[cand.var]++;
                        }
                    }

                    // Recompute score with updated pseudocosts
                    double d_down = pc_down[cand.var] * cand.frac_down;
                    double d_up = pc_up[cand.var] * cand.frac_up;
                    score = (5.0 / 6.0) * std::min(d_down, d_up) + (1.0 / 6.0) * std::max(d_down, d_up);

                    // Bonus: if one side is infeasible, this is a very strong branch
                    if (!sb_down_sol.is_optimal()) score = 1e20;
                    if (!sb_up_sol.is_optimal()) score = 1e20;

                    sb_done++;
                }

                if (score > best_score) {
                    best_score = score;
                    branch_var = cand.var;
                }
            }
        }

        if (branch_var == -1) {
            branch_var = select_most_fractional(child_sol.x, vt, opts.integrality_tol);
        }

        if (branch_var == -1) {
            return;
        }

        double var_val = child_sol.x[branch_var];
        double floor_val = std::floor(var_val);
        double ceil_val = std::ceil(var_val);

        // Child Left: x_j <= floor(x_j)
        double frac_down = var_val - floor_val;
        double frac_up = ceil_val - var_val;

        BnBNode left_child;
        left_child.id = node_counter++;
        left_child.depth = node.depth + 1;
        left_child.lp_bound = child_obj;
        left_child.col_lower = node.col_lower;
        left_child.col_upper = node.col_upper;
        left_child.col_upper[branch_var] = floor_val;
        left_child.col_basis = child_sol.col_basis;
        left_child.row_basis = child_sol.row_basis;
        left_child.branch_var = branch_var;
        left_child.branch_dir = -1;
        left_child.branch_frac_down = frac_down;
        left_child.branch_frac_up = frac_up;
        left_child.parent_obj = child_obj;

        // Child Right: x_j >= ceil(x_j)
        BnBNode right_child;
        right_child.id = node_counter++;
        right_child.depth = node.depth + 1;
        right_child.lp_bound = child_obj;
        right_child.col_lower = node.col_lower;
        right_child.col_upper = node.col_upper;
        right_child.col_lower[branch_var] = ceil_val;
        right_child.col_basis = child_sol.col_basis;
        right_child.row_basis = child_sol.row_basis;
        right_child.branch_var = branch_var;
        right_child.branch_dir = +1;
        right_child.branch_frac_down = frac_down;
        right_child.branch_frac_up = frac_up;
        right_child.parent_obj = child_obj;

        // Push children to queue
        {
            std::lock_guard<std::mutex> lock(tree_mutex);
            if (opts.strategy.node_selection == model::NodeSelection::BestBound) {
                best_first_queue.push(std::move(left_child));
                best_first_queue.push(std::move(right_child));
            } else {
                depth_first_stack.push_back(std::move(left_child));
                depth_first_stack.push_back(std::move(right_child));
            }
        }
    };

    // 6. Tree Search Loop
    int64_t max_nodes = opts.node_limit > 0 ? opts.node_limit : 1000000;
    double time_limit = opts.time_limit_sec;

    // Adaptive in-solve monitor tracking
    int64_t last_check_node = 0;
    double last_check_gap = 1.0;
    auto last_check_time = start_time;
    int strategy_mutations = 0;
    std::string in_solve_log = "";
    double pre_switch_rate = 0.0;
    double post_switch_rate = 0.0;

    auto check_stall_and_adapt = [&]() {
        if (!opts.strategy.enable_in_solve_monitor) return;
        int window = opts.strategy.stall_node_window;
        if (window <= 0) window = 40;
        if (nodes_explored - last_check_node >= window) {
            double cur_gap = 1.0;
            if (has_incumbent && std::abs(best_obj) < 1e20 && std::abs(global_dual_bound) < 1e20) {
                cur_gap = std::abs(best_obj - global_dual_bound) / (1e-10 + std::abs(best_obj));
            }
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - last_check_time).count();
            double cur_rate = (nodes_explored - last_check_node) / (dt > 1e-6 ? dt : 1e-6);

            if (std::abs(last_check_gap - cur_gap) < opts.strategy.stall_gap_tolerance) {
                // Stall detected!
                pre_switch_rate = cur_rate;
                strategy_mutations++;

                // Toggle node selection: BestBound <-> DepthFirst
                if (opts.strategy.node_selection == model::NodeSelection::BestBound) {
                    opts.strategy.node_selection = model::NodeSelection::DepthFirst;
                    while (!best_first_queue.empty()) {
                        depth_first_stack.push_back(std::move(const_cast<BnBNode&>(best_first_queue.top())));
                        best_first_queue.pop();
                    }
                } else {
                    opts.strategy.node_selection = model::NodeSelection::BestBound;
                    for (auto& n : depth_first_stack) {
                        best_first_queue.push(std::move(n));
                    }
                    depth_first_stack.clear();
                }

                // Toggle branching rule: MostFractional <-> PseudoCost
                if (opts.strategy.branching_rule == model::BranchingRule::MostFractional) {
                    opts.strategy.branching_rule = model::BranchingRule::PseudoCost;
                } else {
                    opts.strategy.branching_rule = model::BranchingRule::MostFractional;
                }

                std::string sel_str = (opts.strategy.node_selection == model::NodeSelection::DepthFirst) ? "DepthFirst" : "BestBound";
                std::string br_str = (opts.strategy.branching_rule == model::BranchingRule::PseudoCost) ? "PseudoCost" : "MostFractional";

                in_solve_log += "[InSolveMonitor] Stall detected at node " + std::to_string(nodes_explored.load()) +
                                " (gap=" + std::to_string(cur_gap * 100.0) + "%). " +
                                "Switched strategy to " + sel_str + " + " + br_str +
                                ". Pre-switch node processing rate: " + std::to_string(cur_rate) + " nodes/sec.\n";
            }
            last_check_gap = cur_gap;
            last_check_node = nodes_explored.load();
            last_check_time = now;
        }
    };

    // Check single-threaded vs multithreaded
    if (opts.threads <= 1) {
        // Deterministic single-threaded execution
        while (!stop_flag) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (time_limit > 0.0 && elapsed >= time_limit) {
                time_limit_hit = true;
                break;
            }

            if (nodes_explored >= max_nodes) {
                node_limit_hit = true;
                break;
            }

            BnBNode curr_node;
            bool has_node = false;

            if (opts.strategy.node_selection == model::NodeSelection::BestBound) {
                if (!best_first_queue.empty()) {
                    curr_node = std::move(const_cast<BnBNode&>(best_first_queue.top()));
                    best_first_queue.pop();
                    has_node = true;
                    global_dual_bound = curr_node.lp_bound;
                }
            } else {
                if (!depth_first_stack.empty()) {
                    curr_node = std::move(depth_first_stack.back());
                    depth_first_stack.pop_back();
                    has_node = true;
                }
            }

            if (!has_node) {
                break; // Tree search complete!
            }

            explore_node(std::move(curr_node));
            check_stall_and_adapt();
        }
    } else {
        // Multithreaded execution using custom ThreadPool
        ThreadPool pool(opts.threads);

        while (!stop_flag) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (time_limit > 0.0 && elapsed >= time_limit) {
                time_limit_hit = true;
                break;
            }

            if (nodes_explored >= max_nodes) {
                node_limit_hit = true;
                break;
            }

            BnBNode curr_node;
            bool has_node = false;

            {
                std::lock_guard<std::mutex> lock(tree_mutex);
                check_stall_and_adapt();
                if (opts.strategy.node_selection == model::NodeSelection::BestBound) {
                    if (!best_first_queue.empty()) {
                        curr_node = std::move(const_cast<BnBNode&>(best_first_queue.top()));
                        best_first_queue.pop();
                        has_node = true;
                        global_dual_bound = curr_node.lp_bound;
                    }
                } else {
                    if (!depth_first_stack.empty()) {
                        curr_node = std::move(depth_first_stack.back());
                        depth_first_stack.pop_back();
                        has_node = true;
                    }
                }
            }

            if (!has_node) {
                pool.wait_all();
                std::lock_guard<std::mutex> lock(tree_mutex);
                if (best_first_queue.empty() && depth_first_stack.empty()) {
                    break; // Truly complete
                }
                continue;
            }

            pool.enqueue(explore_node, std::move(curr_node));
        }

        pool.wait_all();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // 7. Format Solution Output
    if (time_limit_hit) {
        final_sol.status = model::SolutionStatus::TimeLimit;
    } else if (node_limit_hit) {
        final_sol.status = model::SolutionStatus::NodeLimit;
    } else if (has_incumbent) {
        final_sol.status = model::SolutionStatus::Optimal;
    } else {
        final_sol.status = model::SolutionStatus::Infeasible;
    }

    if (has_incumbent) {
        final_sol.primal_objective = best_obj;
        final_sol.x = std::move(best_x);
        final_sol.dual_bound = has_incumbent ? std::min(best_obj, global_dual_bound) : global_dual_bound;
        final_sol.mip_gap = std::abs(final_sol.primal_objective - final_sol.dual_bound) /
                            (1e-10 + std::abs(final_sol.primal_objective));
    }

    final_sol.nodes_explored = nodes_explored;
    final_sol.simplex_iterations = total_iterations;
    final_sol.time_wall_sec = total_time;

    if (strategy_mutations > 0) {
        double dt_post = std::chrono::duration<double>(end_time - last_check_time).count();
        int64_t post_nodes = nodes_explored.load() - last_check_node;
        post_switch_rate = (dt_post > 1e-6) ? (post_nodes / dt_post) : pre_switch_rate;
        in_solve_log += "[InSolveMonitor] Post-switch node processing rate: " + std::to_string(post_switch_rate) + " nodes/sec.\n";
    }

    final_sol.strategy_switches = strategy_mutations;
    final_sol.in_solve_log = in_solve_log;

    return final_sol;
}

} // namespace bnb
} // namespace sih
