"""
Training script for MuLoCo-1 optimizer on a language modeling task.

Trains a GPT-style transformer on real text data (OpenWebText / WikiText)
with the MuLoCo-1 optimizer (Muon inner + SGD Nesterov outer).

Usage:
    python train_lm.py                        # Train with MuLoCo-1 on GPU
    python train_lm.py --compare              # Also train vanilla Muon baseline
    python train_lm.py --dataset wikitext      # Use WikiText-103 (smaller download)
    python train_lm.py --steps 5000 --d-model 512 --n-layers 8  # Bigger model
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from muloco.pytorch import MuLoCo1, Muon


# ---------------------------------------------------------------------------
# Model: GPT with SwiGLU FFN and RMSNorm (Gemma3-style)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        ffn_dim = int(d_model * 8 / 3)
        # Round ffn_dim to multiple of 64 for GPU efficiency
        ffn_dim = ((ffn_dim + 63) // 64) * 64
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Dataset: Real text data from HuggingFace
# ---------------------------------------------------------------------------


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset stored as a flat tensor of token IDs."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_dataset_tokens(dataset_name: str, cache_dir: str, seq_len: int):
    """Download and tokenize a real text dataset.

    Returns (train_dataset, val_dataset, vocab_size).
    """
    cache_file = os.path.join(cache_dir, f"{dataset_name}_tokens.pt")
    val_cache_file = os.path.join(cache_dir, f"{dataset_name}_val_tokens.pt")

    if os.path.exists(cache_file) and os.path.exists(val_cache_file):
        print(f"Loading cached tokens from {cache_file}")
        train_tokens = torch.load(cache_file, weights_only=True)
        val_tokens = torch.load(val_cache_file, weights_only=True)
        # tiktoken gpt2 vocab size
        vocab_size = 50257
        return (
            TokenizedDataset(train_tokens, seq_len),
            TokenizedDataset(val_tokens, seq_len),
            vocab_size,
        )

    print(f"Downloading and tokenizing {dataset_name}...")
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50257

    if dataset_name == "openwebtext":
        ds = load_dataset("openwebtext", split="train", cache_dir=cache_dir)
        # Use first 90% for train, last 10% for val
        n = len(ds)
        split_idx = int(n * 0.9)
        train_texts = ds.select(range(split_idx))["text"]
        val_texts = ds.select(range(split_idx, n))["text"]
    elif dataset_name == "wikitext":
        ds_train = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache_dir
        )
        ds_val = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="validation", cache_dir=cache_dir
        )
        train_texts = ds_train["text"]
        val_texts = ds_val["text"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_texts(texts, max_tokens=None):
        all_ids = []
        total = 0
        for text in texts:
            if not text.strip():
                continue
            ids = enc.encode_ordinary(text)
            all_ids.extend(ids)
            total += len(ids)
            if max_tokens and total >= max_tokens:
                break
        return torch.tensor(all_ids[:max_tokens] if max_tokens else all_ids, dtype=torch.long)

    # For openwebtext, limit to ~200M tokens for reasonable download time
    max_train = 200_000_000 if dataset_name == "openwebtext" else None
    max_val = 1_000_000

    print("Tokenizing train split...")
    train_tokens = tokenize_texts(train_texts, max_train)
    print(f"  Train tokens: {len(train_tokens):,}")
    print("Tokenizing val split...")
    val_tokens = tokenize_texts(val_texts, max_val)
    print(f"  Val tokens: {len(val_tokens):,}")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(train_tokens, cache_file)
    torch.save(val_tokens, val_cache_file)
    print(f"Cached tokens to {cache_dir}")

    return (
        TokenizedDataset(train_tokens, seq_len),
        TokenizedDataset(val_tokens, seq_len),
        vocab_size,
    )


# ---------------------------------------------------------------------------
# Parameter group classification
# ---------------------------------------------------------------------------


def create_param_groups(model: nn.Module):
    """
    Classify model parameters into Muon (2D hidden matrices) and AdamW
    (embeddings, norms, head) groups following the MuLoCo paper.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "head" in name:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "algorithm": "muon"})
    if adamw_params:
        groups.append({"params": adamw_params, "algorithm": "adamw"})
    return groups


# ---------------------------------------------------------------------------
# Cosine learning rate schedule with warmup
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine decay to 0.1x base_lr with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer, lr: float):
    """Set learning rate on all param groups."""
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_dataloader, vocab_size, device, max_batches=50):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train_model(
    model: nn.Module,
    optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    vocab_size: int,
    num_steps: int,
    inner_lr: float,
    warmup_steps: int,
    device: torch.device,
    label: str = "MuLoCo-1",
    grad_clip: float = 1.0,
    eval_interval: int = 500,
    log_interval: int = 50,
):
    """Train the model and return training history."""
    model.train()
    data_iter = iter(train_dataloader)
    history = []
    start_time = time.time()

    for step in range(1, num_steps + 1):
        lr = cosine_lr(step, num_steps, warmup_steps, inner_lr)
        set_lr(optimizer, lr)

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        history.append((step, loss_val))

        if step % log_interval == 0 or step == 1:
            avg_loss = sum(l for _, l in history[-log_interval:]) / min(
                len(history), log_interval
            )
            elapsed = time.time() - start_time
            tokens_per_sec = (
                step * x.shape[0] * x.shape[1] / elapsed
            )
            outer_info = ""
            if hasattr(optimizer, "outer_step_count"):
                outer_info = f" | Outer: {optimizer.outer_step_count}"
            print(
                f"[{label}] Step {step:5d} | Loss: {loss_val:.4f} | "
                f"Avg: {avg_loss:.4f} | LR: {lr:.5f}{outer_info} | "
                f"{tokens_per_sec/1e3:.1f}k tok/s | {elapsed:.0f}s",
                flush=True,
            )

        if step % eval_interval == 0:
            val_loss = evaluate(model, val_dataloader, vocab_size, device)
            elapsed = time.time() - start_time
            print(
                f"[{label}] Step {step:5d} | VAL LOSS: {val_loss:.4f} | {elapsed:.0f}s",
                flush=True,
            )

    total_time = time.time() - start_time
    final_avg = sum(l for _, l in history[-100:]) / min(len(history), 100)
    val_loss = evaluate(model, val_dataloader, vocab_size, device)
    print(
        f"\n[{label}] Done: {total_time:.1f}s | "
        f"Train avg(last 100): {final_avg:.4f} | Val: {val_loss:.4f}"
    )
    return history, val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test MuLoCo-1 on language modeling")
    # Data
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=["wikitext", "openwebtext"],
        help="Dataset to use (wikitext is smaller/faster to download)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Cache directory for dataset (default: ./data or $SLURM_TMPDIR/data)",
    )
    # Model
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--n-layers", type=int, default=8, help="Transformer layers")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    # Training
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--inner-lr", type=float, default=0.02, help="Inner Muon LR")
    parser.add_argument("--outer-lr", type=float, default=0.7, help="Outer SGD LR")
    parser.add_argument("--outer-momentum", type=float, default=0.6, help="Outer Nesterov momentum")
    parser.add_argument("--sync-interval", type=int, default=30, help="Sync interval H")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=300, help="LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N steps")
    # Misc
    parser.add_argument("--compare", action="store_true", help="Also train vanilla Muon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir is None:
        slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
        if slurm_tmpdir:
            args.cache_dir = os.path.join(slurm_tmpdir, "data")
        else:
            args.cache_dir = "./data"

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Dataset
    print(f"\nLoading dataset: {args.dataset}")
    train_dataset, val_dataset, vocab_size = load_dataset_tokens(
        args.dataset, args.cache_dir, args.seq_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"Vocab: {vocab_size} | "
        f"Train: {len(train_dataset):,} seqs | "
        f"Val: {len(val_dataset):,} seqs"
    )

    # -----------------------------------------------------------------------
    # Train with MuLoCo-1
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training with MuLoCo-1")
    print("=" * 70)

    torch.manual_seed(args.seed)
    model = GPT(
        vocab_size, args.d_model, args.n_heads, args.n_layers, args.seq_len
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters ({num_params/1e6:.1f}M)")

    param_groups = create_param_groups(model)
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['algorithm']}: {n:,} params")

    muloco_opt = MuLoCo1(
        params=param_groups,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        sync_interval=args.sync_interval,
        weight_decay=args.weight_decay,
    )

    print(f"\nMuLoCo-1 config:")
    print(f"  Inner LR: {args.inner_lr}, Outer LR: {args.outer_lr}")
    print(f"  Outer Momentum: {args.outer_momentum}, Sync H: {args.sync_interval}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(
        f"  Batch: {args.batch_size} x {args.seq_len} = "
        f"{args.batch_size * args.seq_len:,} tokens/step"
    )
    print()

    muloco_hist, muloco_val = train_model(
        model,
        muloco_opt,
        train_dataloader,
        val_dataloader,
        vocab_size,
        args.steps,
        args.inner_lr,
        args.warmup_steps,
        device,
        "MuLoCo-1",
        args.grad_clip,
        args.eval_interval,
        args.log_interval,
    )

    # -----------------------------------------------------------------------
    # Optionally compare with vanilla Muon
    # -----------------------------------------------------------------------
    if args.compare:
        print("\n" + "=" * 70)
        print("Training with vanilla Muon (DP baseline)")
        print("=" * 70)

        torch.manual_seed(args.seed)
        model_bl = GPT(
            vocab_size, args.d_model, args.n_heads, args.n_layers, args.seq_len
        ).to(device)

        if args.compile and hasattr(torch, "compile"):
            model_bl = torch.compile(model_bl)

        muon_opt = Muon(
            params=create_param_groups(model_bl),
            lr=args.inner_lr,
            weight_decay=args.weight_decay,
        )

        muon_hist, muon_val = train_model(
            model_bl,
            muon_opt,
            train_dataloader,
            val_dataloader,
            vocab_size,
            args.steps,
            args.inner_lr,
            args.warmup_steps,
            device,
            "Muon DP",
            args.grad_clip,
            args.eval_interval,
            args.log_interval,
        )

        # Summary
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)
        print(f"  MuLoCo-1 val loss:  {muloco_val:.4f}")
        print(f"  Muon DP  val loss:  {muon_val:.4f}")
        print(f"  Difference: {muon_val - muloco_val:+.4f}")


if __name__ == "__main__":
    main()
