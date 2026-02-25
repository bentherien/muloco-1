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

# Load modules (arrow+thrift BEFORE venv for pyarrow; no internet on compute nodes)
module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0

# Create virtualenv in fast local storage
python -m venv --system-site-packages $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install deps (fast: cached wheels on CVMFS)
pip install --no-index torch tiktoken datasets triton

# Install muloco package
pip install -e ~/scratch/muloco-1

echo "Pip install done."
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())')"

# Pre-cached data (compute nodes have no internet)
export TIKTOKEN_CACHE_DIR=~/scratch/muloco-1/tiktoken_cache
export HF_DATASETS_OFFLINE=1

# Copy pre-downloaded HF dataset cache to fast local storage
mkdir -p $SLURM_TMPDIR/data
cp -r ~/scratch/muloco-1/data/wikitext $SLURM_TMPDIR/data/
echo "Pre-cached HF data copied to $SLURM_TMPDIR/data/"

echo ""
echo "=== Running MuLoCo-1 training ==="
python ~/scratch/muloco-1/examples/train_lm.py \
    --cache-dir $SLURM_TMPDIR/data \
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
