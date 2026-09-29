#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=0-00:10
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=logs/%j-check_gpu.out
#SBATCH --error=logs/%j-check_gpu.err
#
# Ten-second preflight: does JAX actually see the GPU that SLURM allocated?
# Run this after ANY venv change and before submitting a long array. A venv without
# the `cuda13` extra warns to stderr and silently runs ~20x slower on CPU, which is
# indistinguishable from a correct-but-slow job from the outside.

set -uo pipefail

cd ..
PY=${PY:-../.venv/bin/python}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$PY" - <<'PYEOF'
import os, jax, jax.numpy as jnp
backend = jax.default_backend()
print("devices:", jax.devices())
print("backend:", backend)
x = jnp.ones((512, 512))
print("matmul ok, dtype:", (x @ x).dtype)
alloc = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if alloc and backend == "cpu":
    raise SystemExit("FAIL: GPU allocated but JAX is on CPU")
print("PASS" if backend == "gpu" else f"WARN: backend={backend}")
PYEOF
