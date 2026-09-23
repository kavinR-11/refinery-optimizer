#include "sih/utils/affinity.hpp"
#include <iostream>
#include <cassert>

#define ASSERT(expr) \
    if (!(expr)) { \
        std::cerr << "Assertion failed at line " << __LINE__ << ": " #expr << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running test_affinity..." << std::endl;

    int num_procs = sih::utils::get_num_procs();
    std::cout << "Detected logical processors: " << num_procs << std::endl;
    ASSERT(num_procs > 0);

    int cur_core = sih::utils::get_current_core();
    std::cout << "Currently running on core: " << cur_core << std::endl;
    ASSERT(cur_core >= 0);

    // Test pinning to core 0
    bool pinned = sih::utils::pin_thread_to_core(0);
    std::cout << "Pinning to core 0: " << (pinned ? "SUCCESS" : "FAILED") << std::endl;
    ASSERT(pinned);

    // Verify current core is 0
    int new_core = sih::utils::get_current_core();
    std::cout << "Core after pinning: " << new_core << std::endl;
    ASSERT(new_core == 0);

    // Pin to core set {0, 1}
    bool pinned_set = sih::utils::pin_thread_to_cores({0, 1});
    ASSERT(pinned_set);

    std::cout << "test_affinity: PASS" << std::endl;
    return 0;
}
