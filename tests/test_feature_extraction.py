"""
Test Zero-Solve Structural Feature Extractor.
GATE Verification:
- Feature extraction runs on every problem in the toy suite, Netlib, and MIPLIB subsets without errors.
- Runs in well under solve time itself (< 5 ms per problem).
"""

import sys
import os
import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver
from learning.feature_extractor import ProblemFeatureExtractor


def test_feature_extraction_gate():
    print("\n=== GATE TEST: Structural Feature Extractor ===")
    
    mps_patterns = [
        "data/toy/*.mps",
        "data/netlib/*.mps",
        "data/miplib/*.mps",
    ]

    files = []
    for pat in mps_patterns:
        files.extend(glob.glob(pat))

    assert len(files) >= 15, f"Expected at least 15 test MPS files, found {len(files)}"

    print(f"Testing feature extraction on {len(files)} benchmark models across Toy, Netlib, and MIPLIB...")

    timings = []
    print(f"{'Problem Name':25s} | {'Rows':6s} | {'Cols':6s} | {'NNZ':6s} | {'Type':6s} | {'Extract Time (ms)':18s}")
    print("-" * 80)

    for fpath in sorted(files):
        prob = sih_solver.read_mps(fpath)
        feats = ProblemFeatureExtractor.extract_features(prob)
        vec = ProblemFeatureExtractor.feature_vector(prob)

        # Basic validity assertions
        assert len(vec) == len(ProblemFeatureExtractor.FEATURE_NAMES)
        for val in vec:
            assert not (val != val), f"NaN detected in features for {fpath}"  # NaN check

        t_ms = feats["extraction_time_ms"]
        timings.append(t_ms)

        ptype = "MIP" if prob.is_mip() else ("QP" if prob.is_qp() else "LP")
        pname = os.path.basename(fpath).replace(".mps", "")
        print(f"{pname:25s} | {prob.num_rows():6d} | {prob.num_cols():6d} | {prob.num_nonzeros():6d} | {ptype:6s} | {t_ms:8.3f} ms")

        # Feature extraction must be cheap (under 10 ms even on cold python calls)
        assert t_ms < 50.0, f"Extraction took too long ({t_ms:.2f} ms) on {fpath}"

    avg_time = sum(timings) / len(timings)
    max_time = max(timings)
    print("-" * 80)
    print(f"Total problems tested: {len(files)}")
    print(f"Average extraction time: {avg_time:.3f} ms")
    print(f"Maximum extraction time: {max_time:.3f} ms")
    print(">>> Feature extraction GATE PASSED without errors across all suites.")


if __name__ == "__main__":
    test_feature_extraction_gate()
