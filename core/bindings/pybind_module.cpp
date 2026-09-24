#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sih/model/sparse_matrix.hpp"
#include "sih/model/problem.hpp"
#include "sih/model/solution.hpp"
#include "sih/model/options.hpp"
#include "sih/io/mps_io.hpp"
#include "sih/checker/checker.hpp"
#include "sih/telemetry/telemetry.hpp"
#include "sih/utils/affinity.hpp"
#include "sih/simplex/simplex_solver.hpp"
#include "sih/ipm/ipm_solver.hpp"
#include "sih/ipm/crossover.hpp"
#include "sih/bnb/branch_and_bound.hpp"

namespace py = pybind11;
using namespace sih;

PYBIND11_MODULE(_core, m) {
    m.doc() = "SIH 26119 Indigenous Optimization Solver Core C++ Bindings";

    // --- Triplet & SparseMatrix ---
    py::class_<model::Triplet>(m, "Triplet")
        .def(py::init<int64_t, int64_t, double>(), py::arg("row"), py::arg("col"), py::arg("value"))
        .def_readwrite("row", &model::Triplet::row)
        .def_readwrite("col", &model::Triplet::col)
        .def_readwrite("value", &model::Triplet::value);

    py::class_<model::SparseMatrix>(m, "SparseMatrix")
        .def(py::init<int64_t, int64_t>(), py::arg("rows") = 0, py::arg("cols") = 0)
        .def_static("from_triplets", &model::SparseMatrix::from_triplets,
                    py::arg("m"), py::arg("n"), py::arg("triplets"), py::arg("zero_tol") = 1e-15)
        .def("num_rows", &model::SparseMatrix::num_rows)
        .def("num_cols", &model::SparseMatrix::num_cols)
        .def("num_nonzeros", &model::SparseMatrix::num_nonzeros)
        .def("mat_vec", py::overload_cast<const std::vector<double>&>(&model::SparseMatrix::mat_vec, py::const_))
        .def("mat_trans_vec", py::overload_cast<const std::vector<double>&>(&model::SparseMatrix::mat_trans_vec, py::const_))
        .def("transpose", &model::SparseMatrix::transpose)
        .def("get", &model::SparseMatrix::get)
        .def("csc_col_ptr", &model::SparseMatrix::csc_col_ptr)
        .def("csc_row_ind", &model::SparseMatrix::csc_row_ind)
        .def("csc_values", &model::SparseMatrix::csc_values)
        .def("csr_row_ptr", &model::SparseMatrix::csr_row_ptr)
        .def("csr_col_ind", &model::SparseMatrix::csr_col_ind)
        .def("csr_values", &model::SparseMatrix::csr_values)
        .def("is_valid", &model::SparseMatrix::is_valid);

    // --- Enums ---
    py::enum_<model::ObjectiveSense>(m, "ObjectiveSense")
        .value("Minimize", model::ObjectiveSense::Minimize)
        .value("Maximize", model::ObjectiveSense::Maximize)
        .export_values();

    py::enum_<model::VariableType>(m, "VariableType")
        .value("Continuous", model::VariableType::Continuous)
        .value("Binary", model::VariableType::Binary)
        .value("Integer", model::VariableType::Integer)
        .value("SemiContinuous", model::VariableType::SemiContinuous)
        .export_values();

    py::enum_<model::SolutionStatus>(m, "SolutionStatus")
        .value("Optimal", model::SolutionStatus::Optimal)
        .value("Infeasible", model::SolutionStatus::Infeasible)
        .value("Unbounded", model::SolutionStatus::Unbounded)
        .value("TimeLimit", model::SolutionStatus::TimeLimit)
        .value("IterationLimit", model::SolutionStatus::IterationLimit)
        .value("NodeLimit", model::SolutionStatus::NodeLimit)
        .value("NumericalFailure", model::SolutionStatus::NumericalFailure)
        .value("Interrupted", model::SolutionStatus::Interrupted)
        .value("Unknown", model::SolutionStatus::Unknown)
        .export_values();

    py::enum_<model::BasisStatus>(m, "BasisStatus")
        .value("Basic", model::BasisStatus::Basic)
        .value("AtLower", model::BasisStatus::AtLower)
        .value("AtUpper", model::BasisStatus::AtUpper)
        .value("Free", model::BasisStatus::Free)
        .value("Superbasic", model::BasisStatus::Superbasic)
        .export_values();

    // --- Problem ---
    py::class_<model::Problem>(m, "Problem")
        .def(py::init<std::string>(), py::arg("name") = "problem")
        .def("num_rows", &model::Problem::num_rows)
        .def("num_cols", &model::Problem::num_cols)
        .def("num_nonzeros", &model::Problem::num_nonzeros)
        .def("num_quad_nonzeros", &model::Problem::num_quad_nonzeros)
        .def("num_integers", &model::Problem::num_integers)
        .def("num_binaries", &model::Problem::num_binaries)
        .def("is_mip", &model::Problem::is_mip)
        .def("is_qp", &model::Problem::is_qp)
        .def_property("name", &model::Problem::name, &model::Problem::set_name)
        .def_property("sense", &model::Problem::sense, &model::Problem::set_sense)
        .def_property("obj_offset", &model::Problem::obj_offset, &model::Problem::set_obj_offset)
        .def_property("c", py::overload_cast<>(&model::Problem::c, py::const_), &model::Problem::set_c)
        .def_property("A", &model::Problem::A, &model::Problem::set_A)
        .def_property("row_lower", py::overload_cast<>(&model::Problem::row_lower, py::const_),
                      [](model::Problem& p, std::vector<double> v) { p.row_lower() = std::move(v); })
        .def_property("row_upper", py::overload_cast<>(&model::Problem::row_upper, py::const_),
                      [](model::Problem& p, std::vector<double> v) { p.row_upper() = std::move(v); })
        .def_property("col_lower", py::overload_cast<>(&model::Problem::col_lower, py::const_),
                      [](model::Problem& p, std::vector<double> v) { p.col_lower() = std::move(v); })
        .def_property("col_upper", py::overload_cast<>(&model::Problem::col_upper, py::const_),
                      [](model::Problem& p, std::vector<double> v) { p.col_upper() = std::move(v); })
        .def_property("var_types", py::overload_cast<>(&model::Problem::var_types, py::const_),
                      [](model::Problem& p, std::vector<model::VariableType> v) { p.var_types() = std::move(v); })
        .def("has_quadratic", &model::Problem::has_quadratic)
        .def("Q", &model::Problem::Q)
        .def("set_quadratic_objective", &model::Problem::set_quadratic_objective)
        .def("clear_quadratic_objective", &model::Problem::clear_quadratic_objective)
        .def("row_names", &model::Problem::row_names)
        .def("col_names", &model::Problem::col_names)
        .def("set_row_names", &model::Problem::set_row_names)
        .def("set_col_names", &model::Problem::set_col_names)
        .def("get_row_index", &model::Problem::get_row_index)
        .def("get_col_index", &model::Problem::get_col_index)
        .def("has_row", &model::Problem::has_row)
        .def("has_col", &model::Problem::has_col)
        .def("resize", &model::Problem::resize)
        .def("validate", &model::Problem::validate);

    // --- Solution ---
    py::class_<model::Solution>(m, "Solution")
        .def(py::init<>())
        .def_readwrite("status", &model::Solution::status)
        .def_readwrite("x", &model::Solution::x)
        .def_readwrite("slack", &model::Solution::slack)
        .def_readwrite("row_duals", &model::Solution::row_duals)
        .def_readwrite("reduced_costs", &model::Solution::reduced_costs)
        .def_readwrite("col_basis", &model::Solution::col_basis)
        .def_readwrite("row_basis", &model::Solution::row_basis)
        .def_readwrite("primal_objective", &model::Solution::primal_objective)
        .def_readwrite("dual_bound", &model::Solution::dual_bound)
        .def_readwrite("mip_gap", &model::Solution::mip_gap)
        .def_readwrite("simplex_iterations", &model::Solution::simplex_iterations)
        .def_readwrite("barrier_iterations", &model::Solution::barrier_iterations)
        .def_readwrite("nodes_explored", &model::Solution::nodes_explored)
        .def_readwrite("time_wall_sec", &model::Solution::time_wall_sec)
        .def_readwrite("time_cpu_sec", &model::Solution::time_cpu_sec)
        .def_readwrite("time_presolve_sec", &model::Solution::time_presolve_sec)
        .def_readwrite("time_solver_sec", &model::Solution::time_solver_sec)
        .def_readwrite("ray", &model::Solution::ray)
        .def_readwrite("rhs_down", &model::Solution::rhs_down)
        .def_readwrite("rhs_up", &model::Solution::rhs_up)
        .def_readwrite("obj_down", &model::Solution::obj_down)
        .def_readwrite("obj_up", &model::Solution::obj_up)
        .def("is_optimal", &model::Solution::is_optimal)
        .def("is_feasible", &model::Solution::is_feasible);

    // --- Strategy & Options ---
    py::enum_<model::AlgorithmChoice>(m, "AlgorithmChoice")
        .value("Auto", model::AlgorithmChoice::Auto)
        .value("PrimalSimplex", model::AlgorithmChoice::PrimalSimplex)
        .value("DualSimplex", model::AlgorithmChoice::DualSimplex)
        .value("Barrier", model::AlgorithmChoice::Barrier)
        .value("BranchAndBound", model::AlgorithmChoice::BranchAndBound)
        .export_values();

    py::enum_<model::PricingRule>(m, "PricingRule")
        .value("Dantzig", model::PricingRule::Dantzig)
        .value("SteepestEdge", model::PricingRule::SteepestEdge)
        .value("Devex", model::PricingRule::Devex)
        .value("BFRT", model::PricingRule::BFRT)
        .export_values();

    py::enum_<model::BranchingRule>(m, "BranchingRule")
        .value("MostFractional", model::BranchingRule::MostFractional)
        .value("PseudoCost", model::BranchingRule::PseudoCost)
        .value("StrongBranching", model::BranchingRule::StrongBranching)
        .value("Reliability", model::BranchingRule::Reliability)
        .export_values();

    py::enum_<model::NodeSelection>(m, "NodeSelection")
        .value("BestBound", model::NodeSelection::BestBound)
        .value("DepthFirst", model::NodeSelection::DepthFirst)
        .value("BestEstimate", model::NodeSelection::BestEstimate)
        .export_values();

    py::enum_<model::PresolveMode>(m, "PresolveMode")
        .value("Off", model::PresolveMode::Off)
        .value("On", model::PresolveMode::On)
        .value("Aggressive", model::PresolveMode::Aggressive)
        .export_values();

    py::enum_<model::RatioTest>(m, "RatioTest")
        .value("Textbook", model::RatioTest::Textbook)
        .value("Harris", model::RatioTest::Harris)
        .value("HarrisBFRT", model::RatioTest::HarrisBFRT)
        .export_values();

    py::class_<model::StrategyConfig>(m, "StrategyConfig")
        .def(py::init<>())
        .def_readwrite("algorithm", &model::StrategyConfig::algorithm)
        .def_readwrite("pricing_rule", &model::StrategyConfig::pricing_rule)
        .def_readwrite("branching_rule", &model::StrategyConfig::branching_rule)
        .def_readwrite("node_selection", &model::StrategyConfig::node_selection)
        .def_readwrite("presolve", &model::StrategyConfig::presolve)
        .def_readwrite("max_presolve_passes", &model::StrategyConfig::max_presolve_passes)
        .def_readwrite("ratio_test", &model::StrategyConfig::ratio_test)
        .def_readwrite("enable_scaling", &model::StrategyConfig::enable_scaling)
        .def_readwrite("power_of_two_scaling", &model::StrategyConfig::power_of_two_scaling)
        .def_readwrite("refactor_frequency", &model::StrategyConfig::refactor_frequency)
        .def_readwrite("enable_perturbation", &model::StrategyConfig::enable_perturbation)
        .def_readwrite("perturbation_magnitude", &model::StrategyConfig::perturbation_magnitude)
        .def_readwrite("cut_rounds", &model::StrategyConfig::cut_rounds)
        .def_readwrite("enable_gpu", &model::StrategyConfig::enable_gpu)
        .def_readwrite("ipm_max_iterations", &model::StrategyConfig::ipm_max_iterations)
        .def_readwrite("ipm_primal_tol", &model::StrategyConfig::ipm_primal_tol)
        .def_readwrite("ipm_dual_tol", &model::StrategyConfig::ipm_dual_tol)
        .def_readwrite("ipm_gap_tol", &model::StrategyConfig::ipm_gap_tol)
        .def_readwrite("ipm_step_safety", &model::StrategyConfig::ipm_step_safety)
        .def_readwrite("ipm_centering_exponent", &model::StrategyConfig::ipm_centering_exponent)
        .def_readwrite("ipm_enable_crossover", &model::StrategyConfig::ipm_enable_crossover)
        .def_readwrite("ipm_regularization", &model::StrategyConfig::ipm_regularization);

    py::class_<model::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("time_limit_sec", &model::Options::time_limit_sec)
        .def_readwrite("iteration_limit", &model::Options::iteration_limit)
        .def_readwrite("node_limit", &model::Options::node_limit)
        .def_readwrite("threads", &model::Options::threads)
        .def_readwrite("primal_feasibility_tol", &model::Options::primal_feasibility_tol)
        .def_readwrite("dual_feasibility_tol", &model::Options::dual_feasibility_tol)
        .def_readwrite("integrality_tol", &model::Options::integrality_tol)
        .def_readwrite("zero_tol", &model::Options::zero_tol)
        .def_readwrite("log_to_console", &model::Options::log_to_console)
        .def_readwrite("telemetry_log_path", &model::Options::telemetry_log_path)
        .def_readwrite("strategy", &model::Options::strategy);

    // --- Simplex Solver & Sensitivity ---
    py::class_<simplex::SensitivityReport>(m, "SensitivityReport")
        .def(py::init<>())
        .def_readwrite("rhs_down", &simplex::SensitivityReport::rhs_down)
        .def_readwrite("rhs_up", &simplex::SensitivityReport::rhs_up)
        .def_readwrite("obj_down", &simplex::SensitivityReport::obj_down)
        .def_readwrite("obj_up", &simplex::SensitivityReport::obj_up);

    py::class_<simplex::SimplexSolver>(m, "SimplexSolver")
        .def(py::init<>())
        .def_static("solve", &simplex::SimplexSolver::solve,
                    py::arg("problem"), py::arg("options") = model::Options{})
        .def_static("solve_from_basis", &simplex::SimplexSolver::solve_from_basis,
                    py::arg("problem"), py::arg("col_basis"), py::arg("row_basis"),
                    py::arg("options") = model::Options{});

    // --- IPM Solver & Crossover ---
    py::class_<ipm::IpmSolver>(m, "IpmSolver")
        .def(py::init<>())
        .def_static("solve", &ipm::IpmSolver::solve,
                    py::arg("problem"), py::arg("options") = model::Options{});

    py::class_<ipm::Crossover>(m, "Crossover")
        .def(py::init<>())
        .def_static("crossover", &ipm::Crossover::crossover,
                    py::arg("problem"), py::arg("ipm_sol"), py::arg("options") = model::Options{});

    // --- Branch and Bound MILP Solver ---
    py::class_<bnb::BranchAndBound>(m, "BranchAndBoundSolver")
        .def(py::init<>())
        .def_static("solve", &bnb::BranchAndBound::solve,
                    py::arg("problem"), py::arg("options") = model::Options{});

    // --- MPS IO ---
    m.def("read_mps", &io::read_mps, py::arg("filepath"), "Read an MPS or QPS problem file");
    m.def("write_mps", &io::write_mps, py::arg("problem"), py::arg("filepath"), py::arg("free_format") = true,
          "Write problem to MPS/QPS file");

    // --- Checker ---
    py::class_<checker::CheckResult>(m, "CheckResult")
        .def(py::init<>())
        .def_readonly("is_primal_feasible", &checker::CheckResult::is_primal_feasible)
        .def_readonly("max_row_violation", &checker::CheckResult::max_row_violation)
        .def_readonly("max_col_bound_violation", &checker::CheckResult::max_col_bound_violation)
        .def_readonly("max_primal_residual", &checker::CheckResult::max_primal_residual)
        .def_readonly("is_dual_feasible", &checker::CheckResult::is_dual_feasible)
        .def_readonly("max_dual_residual", &checker::CheckResult::max_dual_residual)
        .def_readonly("is_complementary", &checker::CheckResult::is_complementary)
        .def_readonly("max_complementarity_slack", &checker::CheckResult::max_complementarity_slack)
        .def_readonly("is_integrality_satisfied", &checker::CheckResult::is_integrality_satisfied)
        .def_readonly("max_integrality_violation", &checker::CheckResult::max_integrality_violation)
        .def_readonly("evaluated_objective", &checker::CheckResult::evaluated_objective)
        .def_readonly("objective_discrepancy", &checker::CheckResult::objective_discrepancy)
        .def_readonly("is_certificate_valid", &checker::CheckResult::is_certificate_valid)
        .def_readonly("certificate_violation", &checker::CheckResult::certificate_violation)
        .def_readonly("all_checks_passed", &checker::CheckResult::all_checks_passed)
        .def_readonly("summary", &checker::CheckResult::summary);

    m.def("check_solution", &checker::check_solution,
          py::arg("problem"), py::arg("solution"),
          py::arg("tol_primal") = 1e-6, py::arg("tol_dual") = 1e-6, py::arg("tol_int") = 1e-5,
          "Independently verify candidate solution against problem definitions");

    // --- Telemetry ---
    m.def("get_resident_memory_mb", &telemetry::get_resident_memory_mb, "Query resident memory RSS in MB");
    m.def("serialize_solve_telemetry", &telemetry::serialize_solve_telemetry,
          py::arg("problem"), py::arg("solution"), py::arg("options"), py::arg("run_id") = "");
    m.def("log_solve_telemetry", &telemetry::log_solve_telemetry,
          py::arg("problem"), py::arg("solution"), py::arg("options"),
          py::arg("file_path") = "", py::arg("run_id") = "");

    // --- Thread Affinity ---
    m.def("pin_thread_to_core", &utils::pin_thread_to_core, py::arg("core_id"), "Pin thread to CPU core");
    m.def("pin_thread_to_cores", &utils::pin_thread_to_cores, py::arg("core_ids"), "Pin thread to CPU cores");
    m.def("get_current_core", &utils::get_current_core, "Get core index currently running on");
    m.def("get_num_procs", &utils::get_num_procs, "Get number of active logical processors");
}
