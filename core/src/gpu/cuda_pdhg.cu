#ifdef SIH_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace sih {
namespace gpu {

// --- CUDA Kernels ---

// 1. Custom CSR SpMV: y = A * x
__global__ void csr_spmv_kernel(int m,
                               const int* __restrict__ row_ptr,
                               const int* __restrict__ col_ind,
                               const float* __restrict__ values,
                               const float* __restrict__ x,
                               float* __restrict__ y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int k = start; k < end; ++k) {
            sum += values[k] * x[col_ind[k]];
        }
        y[row] = sum;
    }
}

// 2. Custom CSC SpMV: z = A^T * y
__global__ void csc_spmv_kernel(int n,
                               const int* __restrict__ col_ptr,
                               const int* __restrict__ row_ind,
                               const float* __restrict__ values,
                               const float* __restrict__ y,
                               float* __restrict__ z) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        int start = col_ptr[col];
        int end = col_ptr[col + 1];
        for (int k = start; k < end; ++k) {
            sum += values[k] * y[row_ind[k]];
        }
        z[col] = sum;
    }
}

// 3. Primal Gradient Step & Box Projection:
//    x^{k+1} = proj_[l, u](x^k - tau * (c + A^T y))
//    x_bar^{k+1} = 2 x^{k+1} - x^k
__global__ void primal_update_kernel(int n,
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
        float step = tau[j];
        float x_new = x_old - step * grad;

        float low = l[j];
        float upp = u[j];
        if (x_new < low) x_new = low;
        if (x_new > upp) x_new = upp;

        x[j] = x_new;
        x_bar[j] = 2.0f * x_new - x_old;
    }
}

// 4. Dual Gradient Step via Moreau proximal identity:
//    w = A x_bar + y / sigma
//    y^{k+1} = sigma * (w - proj_[row_l, row_u](w))
__global__ void dual_update_kernel(int m,
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

// 5. Restart kernel: resets extrapolation momentum to prevent oscillations
__global__ void restart_kernel(int n, const float* __restrict__ x, float* __restrict__ x_bar) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < n) {
        x_bar[j] = x[j];
    }
}

// 6. Residual Evaluation Kernels
__global__ void row_residuals_kernel(int m,
                                    const float* __restrict__ Ax,
                                    const float* __restrict__ row_l,
                                    const float* __restrict__ row_u,
                                    float* __restrict__ row_viols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < m) {
        float ax = Ax[i];
        float rl = row_l[i];
        float ru = row_u[i];
        float viol = 0.0f;
        if (rl > -1e15f && ax < rl) viol = fmaxf(viol, rl - ax);
        if (ru < 1e15f && ax > ru)  viol = fmaxf(viol, ax - ru);
        row_viols[i] = viol;
    }
}

__global__ void col_residuals_kernel(int n,
                                    const float* __restrict__ x,
                                    const float* __restrict__ c,
                                    const float* __restrict__ AT_y,
                                    const float* __restrict__ l,
                                    const float* __restrict__ u,
                                    const float* __restrict__ tau,
                                    float* __restrict__ col_viols) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < n) {
        float xj = x[j];
        float grad = c[j] + AT_y[j];
        float step = tau[j];
        float proj = xj - step * grad;
        float low = l[j];
        float upp = u[j];
        if (proj < low) proj = low;
        if (proj > upp) proj = upp;
        col_viols[j] = fabsf(xj - proj) / fmaxf(1e-4f, step);
    }
}

// Host Entry Functions

void launch_csr_spmv(int m, const int* d_row_ptr, const int* d_col_ind,
                     const float* d_values, const float* d_x, float* d_y,
                     cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    csr_spmv_kernel<<<grid_size, block_size, 0, stream>>>(m, d_row_ptr, d_col_ind, d_values, d_x, d_y);
}

void launch_csc_spmv(int n, const int* d_col_ptr, const int* d_row_ind,
                     const float* d_values, const float* d_y, float* d_z,
                     cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    csc_spmv_kernel<<<grid_size, block_size, 0, stream>>>(n, d_col_ptr, d_row_ind, d_values, d_y, d_z);
}

void launch_primal_update(int n, const float* d_tau, const float* d_c, const float* d_AT_y,
                          const float* d_l, const float* d_u, float* d_x, float* d_x_bar,
                          cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    primal_update_kernel<<<grid_size, block_size, 0, stream>>>(n, d_tau, d_c, d_AT_y, d_l, d_u, d_x, d_x_bar);
}

void launch_dual_update(int m, const float* d_sigma, const float* d_A_xbar,
                        const float* d_row_l, const float* d_row_u, float* d_y,
                        cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    dual_update_kernel<<<grid_size, block_size, 0, stream>>>(m, d_sigma, d_A_xbar, d_row_l, d_row_u, d_y);
}

void launch_restart(int n, const float* d_x, float* d_x_bar, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    restart_kernel<<<grid_size, block_size, 0, stream>>>(n, d_x, d_x_bar);
}

void launch_residuals(int m, int n,
                      const float* d_Ax, const float* d_row_l, const float* d_row_u, float* d_row_viols,
                      const float* d_x, const float* d_c, const float* d_AT_y, const float* d_l, const float* d_u, const float* d_tau,
                      float* d_col_viols,
                      cudaStream_t stream) {
    int bs = 256;
    int gs_m = (m + bs - 1) / bs;
    row_residuals_kernel<<<gs_m, bs, 0, stream>>>(m, d_Ax, d_row_l, d_row_u, d_row_viols);
    int gs_n = (n + bs - 1) / bs;
    col_residuals_kernel<<<gs_n, bs, 0, stream>>>(n, d_x, d_c, d_AT_y, d_l, d_u, d_tau, d_col_viols);
}

} // namespace gpu
} // namespace sih

#endif // SIH_CUDA_ENABLED
