#include "sih/factorization/amd.hpp"
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <limits>

namespace sih {
namespace factorization {

std::vector<int64_t> AmdOrder::invert_permutation(const std::vector<int64_t>& perm) {
    int64_t n = static_cast<int64_t>(perm.size());
    std::vector<int64_t> inv(n, -1);
    for (int64_t k = 0; k < n; ++k) {
        if (perm[k] >= 0 && perm[k] < n) {
            inv[perm[k]] = k;
        }
    }
    return inv;
}

std::vector<int64_t> AmdOrder::order_matrix(int64_t n,
                                            const std::vector<int64_t>& col_ptr,
                                            const std::vector<int64_t>& row_ind) {
    if (n <= 1) {
        std::vector<int64_t> p(n);
        for (int64_t i = 0; i < n; ++i) p[i] = i;
        return p;
    }

    // 1. Build adjacency list of symmetrized graph G = (V, E)
    std::vector<std::unordered_set<int64_t>> adj(n);
    for (int64_t j = 0; j < n; ++j) {
        int64_t start = col_ptr[j];
        int64_t end   = col_ptr[j + 1];
        for (int64_t k = start; k < end; ++k) {
            int64_t i = row_ind[k];
            if (i != j && i >= 0 && i < n) {
                adj[j].insert(i);
                adj[i].insert(j);
            }
        }
    }

    // 2. Minimum Degree Elimination
    std::vector<int64_t> perm;
    perm.reserve(n);

    std::vector<bool> eliminated(n, false);
    std::vector<int64_t> degree(n);
    for (int64_t i = 0; i < n; ++i) {
        degree[i] = static_cast<int64_t>(adj[i].size());
    }

    for (int64_t step = 0; step < n; ++step) {
        // Find active node with minimum degree
        int64_t min_deg = std::numeric_limits<int64_t>::max();
        int64_t best_p = -1;

        for (int64_t i = 0; i < n; ++i) {
            if (!eliminated[i] && degree[i] < min_deg) {
                min_deg = degree[i];
                best_p = i;
                if (min_deg == 0) break; // Singleton can be eliminated immediately
            }
        }

        if (best_p == -1) {
            // Pick any remaining uneliminated node
            for (int64_t i = 0; i < n; ++i) {
                if (!eliminated[i]) {
                    best_p = i;
                    break;
                }
            }
        }

        eliminated[best_p] = true;
        perm.push_back(best_p);

        // Neighbors of best_p
        std::vector<int64_t> nbrs;
        nbrs.reserve(adj[best_p].size());
        for (int64_t v : adj[best_p]) {
            if (!eliminated[v]) {
                nbrs.push_back(v);
            }
        }

        // Form clique between neighbors
        for (size_t u_idx = 0; u_idx < nbrs.size(); ++u_idx) {
            int64_t u = nbrs[u_idx];
            adj[u].erase(best_p);

            for (size_t v_idx = u_idx + 1; v_idx < nbrs.size(); ++v_idx) {
                int64_t v = nbrs[v_idx];
                adj[u].insert(v);
                adj[v].insert(u);
            }
            degree[u] = static_cast<int64_t>(adj[u].size());
        }
        adj[best_p].clear();
    }

    return perm;
}

std::vector<int64_t> AmdOrder::order_aat(const model::SparseMatrix& A) {
    int64_t m = A.num_rows();
    int64_t n = A.num_cols();
    if (m <= 1) {
        std::vector<int64_t> p(m);
        for (int64_t i = 0; i < m; ++i) p[i] = i;
        return p;
    }

    const auto& col_ptr = A.csc_col_ptr();
    const auto& row_ind = A.csc_row_ind();

    // Construct adjacency of A * A^T
    std::vector<std::unordered_set<int64_t>> adj(m);

    for (int64_t j = 0; j < n; ++j) {
        int64_t start = col_ptr[j];
        int64_t end   = col_ptr[j + 1];

        // Connect all pairs of rows in column j
        for (int64_t k1 = start; k1 < end; ++k1) {
            int64_t r1 = row_ind[k1];
            for (int64_t k2 = k1 + 1; k2 < end; ++k2) {
                int64_t r2 = row_ind[k2];
                if (r1 != r2) {
                    adj[r1].insert(r2);
                    adj[r2].insert(r1);
                }
            }
        }
    }

    // Convert adjacency to CSC pattern
    std::vector<int64_t> aat_col_ptr(m + 1, 0);
    std::vector<int64_t> aat_row_ind;

    for (int64_t i = 0; i < m; ++i) {
        std::vector<int64_t> nbrs(adj[i].begin(), adj[i].end());
        std::sort(nbrs.begin(), nbrs.end());
        for (int64_t nbr : nbrs) {
            aat_row_ind.push_back(nbr);
        }
        aat_col_ptr[i + 1] = static_cast<int64_t>(aat_row_ind.size());
    }

    return order_matrix(m, aat_col_ptr, aat_row_ind);
}

} // namespace factorization
} // namespace sih
