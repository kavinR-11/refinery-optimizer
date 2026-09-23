# Unit tests executable targets
set(TEST_SOURCES
    unit/test_sparse_matrix.cpp
    unit/test_problem_model.cpp
    unit/test_solution_options.cpp
    unit/test_affinity.cpp
    unit/test_mps_io.cpp
    unit/test_checker.cpp
    unit/test_telemetry.cpp
    unit/test_scaler.cpp
    unit/test_sparse_lu.cpp
    unit/test_presolve.cpp
    unit/test_dual_simplex.cpp
    unit/test_amd.cpp
    unit/test_cholesky.cpp
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE sih_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
