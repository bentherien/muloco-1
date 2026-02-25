#!/bin/bash
# MuLoCo-1 test on Alliance Canada cluster
# Run inside a SLURM allocation (srun --pty bash)
set -e

echo "=== Setting up environment ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi 2>/dev/null | head -10

# Load modules (arrow must be loaded BEFORE venv activation for pyarrow)
module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0

# Create virtualenv in SLURM_TMPDIR for fast I/O
VENV=$SLURM_TMPDIR/venv
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv at $VENV..."
    python -m venv --system-site-packages $VENV
fi
source $VENV/bin/activate

# Install dependencies (cached wheels on Compute Canada)
pip install --no-index torch tiktoken datasets

# Install muloco package
pip install -e ~/scratch/muloco-1

echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "=== Starting MuLoCo-1 training ==="
echo "Dataset: wikitext-103"
echo "Model: 512 dim, 8 heads, 8 layers (~50M params)"
echo ""

python ~/scratch/muloco-1/examples/train_lm.py \
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
echo "=== Done ==="
