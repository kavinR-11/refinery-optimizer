#include "sih/utils/affinity.hpp"

#if defined(__linux__) || defined(__linux)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

namespace sih {
namespace utils {

bool pin_thread_to_core(int core_id) {
    if (core_id < 0) return false;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
}

bool pin_thread_to_cores(const std::vector<int>& core_ids) {
    if (core_ids.empty()) return false;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : core_ids) {
        if (core >= 0) {
            CPU_SET(core, &cpuset);
        }
    }
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
}

int get_current_core() {
    return sched_getcpu();
}

int get_num_procs() {
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
}

} // namespace utils
} // namespace sih

#else

namespace sih {
namespace utils {

bool pin_thread_to_core(int) { return false; }
bool pin_thread_to_cores(const std::vector<int>&) { return false; }
int get_current_core() { return -1; }
int get_num_procs() { return 1; }

} // namespace utils
} // namespace sih

#endif
