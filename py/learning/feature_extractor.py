"""
Zero-Solve Structural Feature Extractor for Mathematical Programs.
Computes comprehensive topological, algebraic, and numerical conditioning features
from Problem data in $O(nnz + m + n)$ without solving the problem.
Supports LP, MILP, and QP models.
"""

import math
import time
from typing import Dict, List, Any
import sih_solver


class ProblemFeatureExtractor:
    """
    Extracts structural feature vectors from a sih_solver.Problem instance.
    Features are designed to be fast to compute (< 1 ms on small-medium instances)
    and discriminative for strategy selection.
    """

    FEATURE_NAMES = [
        "log_num_rows",
        "log_num_cols",
        "log_num_nonzeros",
        "density",
        "aspect_ratio",
        "frac_continuous",
        "frac_integer",
        "frac_binary",
        "frac_equality",
        "frac_le",
        "frac_ge",
        "frac_ranged",
        "coeff_dyn_range",
        "coeff_mean",
        "coeff_std",
        "row_deg_mean",
        "row_deg_std",
        "col_deg_mean",
        "col_deg_std",
        "c_density",
        "c_dyn_range",
        "c_norm2",
        "rhs_dyn_range",
        "is_mip",
        "is_qp",
    ]

    @classmethod
    def extract_features(cls, problem: sih_solver.Problem) -> Dict[str, float]:
        """
        Compute structural feature dictionary from a Problem instance.
        """
        t0 = time.perf_counter()

        m = problem.num_rows()
        n = problem.num_cols()
        nnz = problem.num_nonzeros()

        # 1. Size & Sparsity
        log_m = math.log10(max(1, m))
        log_n = math.log10(max(1, n))
        log_nnz = math.log10(max(1, nnz))
        density = (nnz / (m * n)) if (m > 0 and n > 0) else 0.0
        aspect_ratio = math.log10(max(1e-4, m / max(1, n)))

        # 2. Variable Types
        n_int = problem.num_integers()
        n_bin = problem.num_binaries()
        n_cont = max(0, n - n_int - n_bin)
        
        frac_cont = n_cont / max(1, n)
        frac_int = n_int / max(1, n)
        frac_bin = n_bin / max(1, n)

        # 3. Constraint Boundaries
        rl = problem.row_lower
        ru = problem.row_upper
        n_eq = 0
        n_le = 0
        n_ge = 0
        n_ranged = 0

        finite_b = []
        for i in range(m):
            low = rl[i]
            upp = ru[i]
            is_low_inf = (low <= -1e15)
            is_upp_inf = (upp >= 1e15)

            if not is_low_inf and not is_upp_inf:
                if abs(low - upp) < 1e-9:
                    n_eq += 1
                else:
                    n_ranged += 1
                finite_b.append(abs(low))
                finite_b.append(abs(upp))
            elif is_low_inf and not is_upp_inf:
                n_le += 1
                finite_b.append(abs(upp))
            elif not is_low_inf and is_upp_inf:
                n_ge += 1
                finite_b.append(abs(low))

        frac_eq = n_eq / max(1, m)
        frac_le = n_le / max(1, m)
        frac_ge = n_ge / max(1, m)
        frac_ranged = n_ranged / max(1, m)

        # 4. Matrix Nonzeros Distribution
        vals = problem.A.csc_values()
        col_ptr = problem.A.csc_col_ptr()
        row_ind = problem.A.csc_row_ind()

        abs_vals = [abs(v) for v in vals if abs(v) > 1e-15]
        if abs_vals:
            c_min = min(abs_vals)
            c_max = max(abs_vals)
            coeff_dyn_range = math.log10(max(1e-12, c_max / max(1e-15, c_min)))
            coeff_mean = sum(abs_vals) / len(abs_vals)
            coeff_std = math.sqrt(sum((v - coeff_mean) ** 2 for v in abs_vals) / len(abs_vals))
        else:
            coeff_dyn_range = 0.0
            coeff_mean = 0.0
            coeff_std = 0.0

        # Degree statistics
        row_counts = [0] * max(1, m)
        for r in row_ind:
            if r < m:
                row_counts[r] += 1
        
        row_deg_mean = sum(row_counts) / max(1, m)
        row_deg_std = math.sqrt(sum((rc - row_deg_mean) ** 2 for rc in row_counts) / max(1, m))

        col_counts = [col_ptr[j + 1] - col_ptr[j] for j in range(n)] if len(col_ptr) > n else [0]
        col_deg_mean = sum(col_counts) / max(1, n)
        col_deg_std = math.sqrt(sum((cc - col_deg_mean) ** 2 for cc in col_counts) / max(1, n))

        # 5. Objective Vector Conditioning
        c_vals = problem.c
        abs_c = [abs(v) for v in c_vals if abs(v) > 1e-15]
        c_density = len(abs_c) / max(1, n)
        if abs_c:
            c_min_val = min(abs_c)
            c_max_val = max(abs_c)
            c_dyn_range = math.log10(max(1e-12, c_max_val / max(1e-15, c_min_val)))
            c_norm2 = math.sqrt(sum(v * v for v in c_vals))
        else:
            c_dyn_range = 0.0
            c_norm2 = 0.0

        # 6. RHS Conditioning
        pos_b = [v for v in finite_b if v > 1e-15]
        if pos_b:
            b_min_val = min(pos_b)
            b_max_val = max(pos_b)
            rhs_dyn_range = math.log10(max(1e-12, b_max_val / max(1e-15, b_min_val)))
        else:
            rhs_dyn_range = 0.0

        t1 = time.perf_counter()

        features = {
            "log_num_rows": log_m,
            "log_num_cols": log_n,
            "log_num_nonzeros": log_nnz,
            "density": density,
            "aspect_ratio": aspect_ratio,
            "frac_continuous": frac_cont,
            "frac_integer": frac_int,
            "frac_binary": frac_bin,
            "frac_equality": frac_eq,
            "frac_le": frac_le,
            "frac_ge": frac_ge,
            "frac_ranged": frac_ranged,
            "coeff_dyn_range": coeff_dyn_range,
            "coeff_mean": coeff_mean,
            "coeff_std": coeff_std,
            "row_deg_mean": row_deg_mean,
            "row_deg_std": row_deg_std,
            "col_deg_mean": col_deg_mean,
            "col_deg_std": col_deg_std,
            "c_density": c_density,
            "c_dyn_range": c_dyn_range,
            "c_norm2": c_norm2,
            "rhs_dyn_range": rhs_dyn_range,
            "is_mip": 1.0 if problem.is_mip() else 0.0,
            "is_qp": 1.0 if problem.is_qp() else 0.0,
            "extraction_time_ms": (t1 - t0) * 1000.0,
        }

        return features

    @classmethod
    def feature_vector(cls, problem: sih_solver.Problem) -> List[float]:
        """
        Return ordered numerical feature list matching FEATURE_NAMES.
        """
        feats = cls.extract_features(problem)
        return [feats[name] for name in cls.FEATURE_NAMES]
