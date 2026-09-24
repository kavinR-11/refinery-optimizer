/*
 * Baseline reference only (Hard Rule 1 amendment: cuSPARSE permitted ONLY under /bench).
 * Benchmarks cuSPARSE SpMV performance vs our custom CSR/CSC kernels.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <sih/model/problem.hpp>
#include <sih/io/mps_io.hpp>

// Primal projection kernel for cuSPARSE baseline
__global__ void cusparse_primal_update_kernel(int n,
                                              const float* __restrict__ tau,
                                              const float* __restrict__ c,
                                              const float* __restrict__ AT_y,
                                              const float* __restrict__ l,
                                              const float* __restrict__ u,
                                              float* __restrict__ x,
                                              float* __restrict__ x_bar) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < n) {
        float x_old = x[j];
        float grad = c[j] + AT_y[j];
        float x_new = x_old - tau[j] * grad;
        float low = l[j];
        float upp = u[j];
        if (x_new < low) x_new = low;
        if (x_new > upp) x_new = upp;
        x[j] = x_new;
        x_bar[j] = 2.0f * x_new - x_old;
    }
}

// Dual projection kernel
__global__ void cusparse_dual_update_kernel(int m,
                                            const float* __restrict__ sigma,
                                            const float* __restrict__ A_xbar,
                                            const float* __restrict__ row_l,
                                            const float* __restrict__ row_u,
                                            float* __restrict__ y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < m) {
        float sig = sigma[i];
        float y_old = y[i];
        float ax = A_xbar[i];
        float rl = row_l[i];
        float ru = row_u[i];
        float w = ax + (y_old / sig);
        float p = w;
        if (p < rl) p = rl;
        if (p > ru) p = ru;
        y[i] = sig * (w - p);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mps_file> [iterations=1000]\n";
        return 1;
    }

    std::string mps_path = argv[1];
    int max_iters = (argc >= 3) ? std::atoi(argv[2]) : 1000;

    auto prob = sih::io::read_mps(mps_path);
    int m = static_cast<int>(prob.num_rows());
    int n = static_cast<int>(prob.num_cols());
    int nnz = static_cast<int>(prob.num_nonzeros());

    // Build CSR
    const auto& csc_mat = prob.A();
    const auto& csc_col_ptr = csc_mat.csc_col_ptr();
    const auto& csc_row_ind = csc_mat.csc_row_ind();
    const auto& csc_vals_d = csc_mat.csc_values();

    std::vector<int> csr_row_ptr(m + 1, 0);
    for (int idx : csc_row_ind) {
        if (idx < m) csr_row_ptr[idx + 1]++;
    }
    for (int i = 0; i < m; ++i) csr_row_ptr[i + 1] += csr_row_ptr[i];

    std::vector<int> csr_col_ind(nnz);
    std::vector<float> csr_vals_f(nnz);
    std::vector<int> row_fill = csr_row_ptr;

    for (int c = 0; c < n; ++c) {
        for (int idx = csc_col_ptr[c]; idx < csc_col_ptr[c + 1]; ++idx) {
            int r = csc_row_ind[idx];
            int dest = row_fill[r]++;
            csr_col_ind[dest] = c;
            csr_vals_f[dest] = static_cast<float>(csc_vals_d[idx]);
        }
    }

    std::vector<float> h_c(n), h_l(n), h_u(n), h_rl(m), h_ru(m);
    std::vector<float> h_tau(n), h_sigma(m);

    for (int j = 0; j < n; ++j) {
        h_c[j] = static_cast<float>(prob.c()[j]);
        h_l[j] = static_cast<float>(prob.col_lower()[j]);
        h_u[j] = static_cast<float>(prob.col_upper()[j]);
        float col_sum = 0.0f;
        for (int k = csc_col_ptr[j]; k < csc_col_ptr[j + 1]; ++k) {
            col_sum += std::abs(static_cast<float>(csc_vals_d[k]));
        }
        h_tau[j] = 0.95f / std::max(1.0f, col_sum);
    }

    for (int i = 0; i < m; ++i) {
        h_rl[i] = static_cast<float>(prob.row_lower()[i]);
        h_ru[i] = static_cast<float>(prob.row_upper()[i]);
        float row_sum = 0.0f;
        for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k) {
            row_sum += std::abs(csr_vals_f[k]);
        }
        h_sigma[i] = 0.95f / std::max(1.0f, row_sum);
    }

    // Allocate GPU buffers
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_ind = nullptr;
    float* d_csr_vals = nullptr;
    float *d_c = nullptr, *d_l = nullptr, *d_u = nullptr, *d_rl = nullptr, *d_ru = nullptr;
    float *d_tau = nullptr, *d_sigma = nullptr;
    float *d_x = nullptr, *d_x_bar = nullptr, *d_y = nullptr;
    float *d_AT_y = nullptr, *d_A_xbar = nullptr;

    cudaMalloc(&d_csr_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_ind, nnz * sizeof(int));
    cudaMalloc(&d_csr_vals, nnz * sizeof(float));

    cudaMalloc(&d_c, n * sizeof(float));
    cudaMalloc(&d_l, n * sizeof(float));
    cudaMalloc(&d_u, n * sizeof(float));
    cudaMalloc(&d_rl, m * sizeof(float));
    cudaMalloc(&d_ru, m * sizeof(float));
    cudaMalloc(&d_tau, n * sizeof(float));
    cudaMalloc(&d_sigma, m * sizeof(float));

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_x_bar, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMalloc(&d_AT_y, n * sizeof(float));
    cudaMalloc(&d_A_xbar, m * sizeof(float));

    cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_vals, csr_vals_f.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl, h_rl.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ru, h_ru.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, h_tau.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, h_sigma.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_x(n, 0.0f);
    for (int j = 0; j < n; ++j) h_x[j] = std::max(h_l[j], std::min(h_u[j], 0.0f));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_bar, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, m * sizeof(float));

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, m, n, nnz,
                      d_csr_row_ptr, d_csr_col_ind, d_csr_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecXbar, vecY, vecATy, vecAxbar;
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecXbar, n, d_x_bar, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_32F);
    cusparseCreateDnVec(&vecATy, n, d_AT_y, CUDA_R_32F);
    cusparseCreateDnVec(&vecAxbar, m, d_A_xbar, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_fwd = 0, buf_bwd = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecXbar, &beta, vecAxbar,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_fwd);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                            &alpha, matA, vecY, &beta, vecATy,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_bwd);

    size_t max_buf = std::max(buf_fwd, buf_bwd);
    void* d_buffer = nullptr;
    if (max_buf > 0) cudaMalloc(&d_buffer, max_buf);

    int bs = 256;
    int gs_n = (n + bs - 1) / bs;
    int gs_m = (m + bs - 1) / bs;

    // Run timed iterations
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. cuSPARSE A^T y
        cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                     &alpha, matA, vecY, &beta, vecATy,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

        // 2. Primal update
        cusparse_primal_update_kernel<<<gs_n, bs>>>(n, d_tau, d_c, d_AT_y, d_l, d_u, d_x, d_x_bar);

        // 3. cuSPARSE A x_bar
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecXbar, &beta, vecAxbar,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

        // 4. Dual update
        cusparse_dual_update_kernel<<<gs_m, bs>>>(m, d_sigma, d_A_xbar, d_rl, d_ru, d_y);
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Copy back objective
    cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    double obj = 0.0;
    for (int j = 0; j < n; ++j) obj += prob.c()[j] * h_x[j];

    std::cout << "[cuSPARSE Baseline] Iters=" << max_iters
              << " Time=" << total_ms << " ms"
              << " (" << (total_ms / max_iters) * 1000.0 << " us/iter)"
              << " Obj=" << obj << "\n";

    // Cleanup
    if (d_buffer) cudaFree(d_buffer);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecXbar);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecATy);
    cusparseDestroyDnVec(vecAxbar);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

    cudaFree(d_csr_row_ptr); cudaFree(d_csr_col_ind); cudaFree(d_csr_vals);
    cudaFree(d_c); cudaFree(d_l); cudaFree(d_u); cudaFree(d_rl); cudaFree(d_ru);
    cudaFree(d_tau); cudaFree(d_sigma);
    cudaFree(d_x); cudaFree(d_x_bar); cudaFree(d_y);
    cudaFree(d_AT_y); cudaFree(d_A_xbar);

    return 0;
}
