#pragma once

#include <vector>

namespace sih {
namespace utils {

// Pin calling thread to a single core using sched_setaffinity
bool pin_thread_to_core(int core_id);

// Pin calling thread to a set of cores
bool pin_thread_to_cores(const std::vector<int>& core_ids);

// Query core on which the calling thread is currently executing (sched_getcpu)
int get_current_core();

// Query number of logical cores available to the system
int get_num_procs();

} // namespace utils
} // namespace sih
