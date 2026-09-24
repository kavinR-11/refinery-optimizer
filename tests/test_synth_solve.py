import sih_solver
import time

path = "data/synthetic/synth_500x1000.mps"
prob = sih_solver.read_mps(path)
print(f"Loaded {prob.name}: Rows={prob.num_rows()}, Cols={prob.num_cols()}, Nonzeros={prob.num_nonzeros()}")

# CPU Simplex solve
t0 = time.perf_counter()
sol_cpu = sih_solver.solve(prob)
t_cpu = (time.perf_counter() - t0) * 1000.0
print(f"CPU Simplex: status={sol_cpu.status}, obj={sol_cpu.primal_objective:.6f}, time={t_cpu:.2f} ms")

# GPU Hybrid solve
cfg = sih_solver.GpuPdhgConfig()
cfg.max_iterations = 1000
cfg.check_frequency = 25
t0 = time.perf_counter()
sol_hyb = sih_solver.solve_gpu_hybrid(prob, cfg)
t_hyb = (time.perf_counter() - t0) * 1000.0
print(f"GPU Hybrid : status={sol_hyb.status}, obj={sol_hyb.primal_objective:.6f}, time={t_hyb:.2f} ms")

rel_diff = abs(sol_hyb.primal_objective - sol_cpu.primal_objective) / max(1.0, abs(sol_cpu.primal_objective))
print(f"Relative obj diff: {rel_diff:.2e}")
assert rel_diff < 1e-3, f"Objective mismatch: {rel_diff}"
print("Synthetic solve test passed!")
