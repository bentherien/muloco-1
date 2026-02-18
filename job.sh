#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=muloco1
#SBATCH --output=%x_%j.out

set -e

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date)"
nvidia-smi -L

# Load modules
module load python/3.11.5 cuda/12.6

# Create virtualenv in fast local storage
python -m venv --system-site-packages $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install deps (fast: cached wheels on CVMFS)
pip install --no-index torch tiktoken datasets 2>&1 | tail -3
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())')"

# Copy repo to fast local storage
cp -r ~/scratch/muloco-1 $SLURM_TMPDIR/muloco-1
cd $SLURM_TMPDIR/muloco-1

echo ""
echo "=== Running MuLoCo-1 training ==="
python test_muloco.py \
    --dataset wikitext \
    --d-model 512 \
    --n-heads 8 \
    --n-layers 8 \
    --seq-len 1024 \
    --batch-size 64 \
    --steps 5000 \
    --inner-lr 0.02 \
    --outer-lr 0.7 \
    --outer-momentum 0.6 \
    --sync-interval 30 \
    --weight-decay 0.01 \
    --warmup-steps 300 \
    --eval-interval 500 \
    --log-interval 50 \
    --compare \
    --compile

echo ""
echo "=== Done: $(date) ==="
