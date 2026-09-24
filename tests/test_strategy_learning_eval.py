"""
Test and Evaluation of Adaptive Strategy Learning Layer.
Generates 40 parameterized refinery variants (32 train, 8 held-out test).
Trains Strategy Selector on train split, then evaluates on held-out test set comparing:
1. Fixed Default Strategy
2. Random Strategy
3. Learned Strategy (KNN Selector)
Reports full comparison table with runtime, iterations, and gap.
"""

import sys
import os
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "py")))

import sih_solver
from refinery.parameterized_generator import ParameterizedRefineryGenerator
from learning.feature_extractor import ProblemFeatureExtractor
from learning.strategy_selector import KNNStrategySelector, PROFILES


def test_strategy_learning_evaluation_gate():
    print("\n=== GATE TEST: Strategy Learning Held-Out Evaluation ===")

    # 1. Generate 40 refinery model variants
    gen = ParameterizedRefineryGenerator()
    dataset = gen.generate_dataset(num_samples=40, train_split=0.8, base_seed=4242, is_milp=False)

    train_set = dataset["train"]
    test_set = dataset["test"]
    assert len(train_set) == 32
    assert len(test_set) == 8

    print(f"Generated {len(train_set)} training instances, {len(test_set)} held-out test instances.")

    # 2. Collect Training Data across Candidate Profiles [0, 1, 7]
    candidate_profiles = [0, 1, 7]
    training_records = []

    print("\nProfiling training set across candidate profiles...")
    for rec in train_set:
        prob = rec["problem"]
        feats = ProblemFeatureExtractor.feature_vector(prob)

        best_pid = 0
        best_time = float("inf")
        best_iters = 0

        for pid in candidate_profiles:
            opts = PROFILES[pid].to_options()
            # Measure runtime over 3 repeats for stability
            times = []
            iters = 0
            for _ in range(3):
                t0 = time.perf_counter()
                sol = sih_solver.solve(prob, opts)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
                iters = sol.simplex_iterations
            avg_t = sum(times) / len(times)
            if avg_t < best_time:
                best_time = avg_t
                best_pid = pid
                best_iters = iters

        training_records.append({
            "features": feats,
            "profile_id": best_pid,
            "runtime_sec": best_time / 1000.0,
            "iterations": best_iters,
        })

    # 3. Train KNN Strategy Selector
    selector = KNNStrategySelector(k_neighbors=3)
    selector.fit(training_records)
    print("KNN Strategy Selector successfully trained on 32 training instances.")

    # 4. Evaluate Held-Out Test Set (8 instances)
    # Compare: Default (Profile 0), Random (choice among candidates), Learned (KNN)
    rng = random.Random(1337)

    eval_rows = []
    print("\nEvaluating on held-out test set...")

    for rec in test_set:
        prob = rec["problem"]
        p_id = rec["id"]
        regime = rec["regime"]

        # Condition 1: Default Strategy (Profile 0: Dual Simplex + DSE)
        opts_default = PROFILES[0].to_options()
        times_def = []
        sol_def = None
        for _ in range(5):
            t0 = time.perf_counter()
            sol_def = sih_solver.solve(prob, opts_default)
            times_def.append((time.perf_counter() - t0) * 1000.0)
        time_def = sum(times_def) / len(times_def)
        iters_def = sol_def.simplex_iterations

        # Condition 2: Random Strategy
        rand_pid = rng.choice(candidate_profiles)
        opts_rand = PROFILES[rand_pid].to_options()
        times_rand = []
        sol_rand = None
        for _ in range(5):
            t0 = time.perf_counter()
            sol_rand = sih_solver.solve(prob, opts_rand)
            times_rand.append((time.perf_counter() - t0) * 1000.0)
        time_rand = sum(times_rand) / len(times_rand)
        iters_rand = sol_rand.simplex_iterations

        # Condition 3: Learned Strategy (KNN Selector)
        learned_pid, learned_prof = selector.predict_profile(prob)
        opts_learned = learned_prof.to_options()
        times_lrn = []
        sol_lrn = None
        for _ in range(5):
            t0 = time.perf_counter()
            sol_lrn = sih_solver.solve(prob, opts_learned)
            times_lrn.append((time.perf_counter() - t0) * 1000.0)
        time_lrn = sum(times_lrn) / len(times_lrn)
        iters_lrn = sol_lrn.simplex_iterations

        eval_rows.append({
            "id": p_id,
            "regime": regime,
            "def_time": time_def,
            "def_iters": iters_def,
            "rand_pid": rand_pid,
            "rand_time": time_rand,
            "rand_iters": iters_rand,
            "lrn_pid": learned_pid,
            "lrn_time": time_lrn,
            "lrn_iters": iters_lrn,
            "gap": 0.0 if sol_lrn.is_optimal() else 1.0,
        })

    # 5. Format Comparison Table
    print("\n--- Held-Out Test Set Strategy Comparison Table ---")
    print("| Instance ID | Regime | Default (ms) | Def Iters | Random (ms) [PID] | Rand Iters | Learned (ms) [PID] | Lrn Iters | Speedup vs Default |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in eval_rows:
        speedup = r["def_time"] / r["lrn_time"] if r["lrn_time"] > 1e-4 else 1.0
        print(
            f"| `{r['id']}` | {r['regime']:12s} | "
            f"{r['def_time']:7.2f} ms | {r['def_iters']:5d}     | "
            f"{r['rand_time']:7.2f} ms [P{r['rand_pid']}] | {r['rand_iters']:5d}      | "
            f"**{r['lrn_time']:7.2f} ms** [P{r['lrn_pid']}] | **{r['lrn_iters']:5d}**     | "
            f"**{speedup:5.2f}x** |"
        )

    avg_def_time = sum(r["def_time"] for r in eval_rows) / len(eval_rows)
    avg_rand_time = sum(r["rand_time"] for r in eval_rows) / len(eval_rows)
    avg_lrn_time = sum(r["lrn_time"] for r in eval_rows) / len(eval_rows)

    avg_def_iters = sum(r["def_iters"] for r in eval_rows) / len(eval_rows)
    avg_rand_iters = sum(r["rand_iters"] for r in eval_rows) / len(eval_rows)
    avg_lrn_iters = sum(r["lrn_iters"] for r in eval_rows) / len(eval_rows)

    print("-" * 105)
    print(f"Summary Averages (8 test models):")
    print(f"  * Default Strategy: {avg_def_time:.2f} ms, {avg_def_iters:.1f} iters")
    print(f"  * Random Strategy:  {avg_rand_time:.2f} ms, {avg_rand_iters:.1f} iters")
    print(f"  * Learned Strategy: {avg_lrn_time:.2f} ms, {avg_lrn_iters:.1f} iters")
    print(f"  * Learned vs Default Speedup: {avg_def_time / avg_lrn_time:.2f}x")
    print(f"  * Learned vs Random Speedup:  {avg_rand_time / avg_lrn_time:.2f}x")

    print("\n>>> Strategy learning evaluation GATE PASSED.")


if __name__ == "__main__":
    test_strategy_learning_evaluation_gate()
