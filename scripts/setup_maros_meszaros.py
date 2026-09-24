import shutil
from pathlib import Path

src_dir = Path("/tmp/maros-meszaros")
dst_dir = Path("/home/notanracistfr/projects/sih26119-solver/data/maros_meszaros")
dst_dir.mkdir(parents=True, exist_ok=True)

names = [
    "CVXQP1_S",
    "CVXQP2_S",
    "CVXQP3_S",
    "DUAL1",
    "DUAL2",
    "HS21",
    "HS35",
    "HS51",
    "HS52",
    "HS76",
]

for name in names:
    src_file = src_dir / f"{name}.SIF"
    if src_file.exists():
        dst_file = dst_dir / f"{name}.mps"
        shutil.copy(src_file, dst_file)
        print(f"Copied {name}.SIF -> {dst_file.name}")
    else:
        print(f"Warning: {src_file} does not exist")
