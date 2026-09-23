#include "sih/telemetry/telemetry.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <unistd.h>

namespace sih {
namespace telemetry {

double get_resident_memory_mb() {
#if defined(__linux__) || defined(__linux)
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size = 0, resident = 0;
        statm >> size >> resident;
        long page_size = sysconf(_SC_PAGESIZE);
        return (resident * page_size) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

namespace {

std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
    gmtime_r(&timer, &bt);

    std::ostringstream ss;
    ss << std::put_time(&bt, "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return ss.str();
}

std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    uint64_t part1 = dis(gen);
    uint64_t part2 = dis(gen);

    std::ostringstream ss;
    ss << std::hex << std::setfill('0')
       << std::setw(8) << (part1 >> 32) << "-"
       << std::setw(4) << ((part1 >> 16) & 0xFFFF) << "-"
       << std::setw(4) << (part1 & 0xFFFF) << "-"
       << std::setw(4) << (part2 >> 48) << "-"
       << std::setw(12) << (part2 & 0xFFFFFFFFFFFFULL);
    return ss.str();
}

} // anonymous namespace

std::string serialize_solve_telemetry(const model::Problem& problem,
                                     const model::Solution& solution,
                                     const model::Options& options,
                                     const std::string& run_id) {
    std::string actual_run_id = run_id.empty() ? generate_uuid() : run_id;
    std::string timestamp = get_iso_timestamp();

    int64_t m = problem.num_rows();
    int64_t n = problem.num_cols();
    int64_t nnz = problem.num_nonzeros();
    double density = (m > 0 && n > 0) ? static_cast<double>(nnz) / (m * n) : 0.0;

    std::ostringstream ss;
    ss << std::setprecision(10);
    ss << "{"
       << "\"timestamp\":\"" << timestamp << "\","
       << "\"run_id\":\"" << actual_run_id << "\","
       << "\"problem\":{"
       << "\"name\":\"" << problem.name() << "\","
       << "\"sense\":\"" << (problem.sense() == model::ObjectiveSense::Minimize ? "minimize" : "maximize") << "\","
       << "\"num_rows\":" << m << ","
       << "\"num_cols\":" << n << ","
       << "\"num_nonzeros\":" << nnz << ","
       << "\"num_quad_nonzeros\":" << problem.num_quad_nonzeros() << ","
       << "\"num_integers\":" << problem.num_integers() << ","
       << "\"num_binaries\":" << problem.num_binaries() << ","
       << "\"density\":" << density
       << "},"
       << "\"strategy\":{"
       << "\"algorithm\":\"" << model::algorithm_to_string(options.strategy.algorithm) << "\","
       << "\"pricing_rule\":\"" << model::pricing_to_string(options.strategy.pricing_rule) << "\","
       << "\"branching_rule\":\"" << model::branching_to_string(options.strategy.branching_rule) << "\","
       << "\"node_selection\":\"" << model::node_selection_to_string(options.strategy.node_selection) << "\","
       << "\"presolve\":\"" << model::presolve_to_string(options.strategy.presolve) << "\","
       << "\"cut_rounds\":" << options.strategy.cut_rounds << ","
       << "\"enable_gpu\":" << (options.strategy.enable_gpu ? "true" : "false") << ","
       << "\"threads\":" << options.threads
       << "},"
       << "\"tolerances\":{"
       << "\"primal_feasibility\":" << options.primal_feasibility_tol << ","
       << "\"dual_feasibility\":" << options.dual_feasibility_tol << ","
       << "\"integrality\":" << options.integrality_tol
       << "},"
       << "\"metrics\":{"
       << "\"status\":\"" << model::status_to_string(solution.status) << "\","
       << "\"primal_objective\":" << solution.primal_objective << ","
       << "\"dual_bound\":" << solution.dual_bound << ","
       << "\"mip_gap\":" << solution.mip_gap << ","
       << "\"simplex_iterations\":" << solution.simplex_iterations << ","
       << "\"barrier_iterations\":" << solution.barrier_iterations << ","
       << "\"nodes_explored\":" << solution.nodes_explored << ","
       << "\"runtime_wall_ms\":" << (solution.time_wall_sec * 1000.0) << ","
       << "\"runtime_cpu_ms\":" << (solution.time_cpu_sec * 1000.0) << ","
       << "\"memory_rss_mb\":" << get_resident_memory_mb()
       << "}"
       << "}";

    return ss.str();
}

bool log_solve_telemetry(const model::Problem& problem,
                         const model::Solution& solution,
                         const model::Options& options,
                         const std::string& file_path,
                         const std::string& run_id) {
    std::string path = file_path.empty() ? options.telemetry_log_path : file_path;
    if (path.empty()) return false;

    std::string json_line = serialize_solve_telemetry(problem, solution, options, run_id);

    std::ofstream out(path, std::ios_base::app);
    if (!out.is_open()) return false;

    out << json_line << "\n";
    return true;
}

} // namespace telemetry
} // namespace sih
