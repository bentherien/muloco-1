# MuLoCo: Alternative Implementation Plans

This document outlines alternative approaches that were considered for the MuLoCo package architecture, along with trade-offs for each.

## Current Approach: Separate Subpackages

**Structure:** `muloco.pytorch` and `muloco.jax` as independent subpackages with optional dependencies.

**Advantages:**
- Clean separation of framework-specific code
- Neither framework is required for installation
- Import errors are clear and actionable
- Each subpackage can evolve independently

**Trade-offs:**
- Slightly longer import paths (`from muloco.pytorch import MuLoCo1` vs `from muloco import MuLoCo1`)
- Two implementations to maintain for algorithm changes

---

## Alternative 1: Unified Interface with Backend Dispatch

```python
from muloco import MuLoCo, set_backend

set_backend("pytorch")  # or "jax"
optimizer = MuLoCo(params, inner_lr=0.02, outer_lr=0.7)
```

A single `MuLoCo` class that dispatches to the appropriate backend based on a global or per-call setting. The interface would be framework-agnostic.

**Advantages:**
- Single import for all users
- Framework-agnostic training scripts
- Easier to switch between frameworks during experimentation

**Trade-offs:**
- Leaky abstraction: PyTorch optimizers and optax GradientTransformations have fundamentally different APIs (`optimizer.step()` vs `opt.update(grads, state, params)`)
- The abstraction layer adds complexity without real benefit since users already commit to one framework
- Harder to expose framework-specific features (e.g., `torch.compile`, optax schedules)
- Would require wrapping optax's functional API to look stateful, or vice versa

**Verdict:** Over-engineered for the current use case. The two frameworks have different enough APIs that a thin wrapper would either be incomplete or add unnecessary indirection.

---

## Alternative 2: Monorepo with Separate Packages

```
muloco/
├── muloco-pytorch/
│   ├── pyproject.toml
│   └── muloco_pytorch/
│       └── ...
├── muloco-jax/
│   ├── pyproject.toml
│   └── muloco_jax/
│       └── ...
└── README.md
```

Two separate pip packages (`muloco-pytorch`, `muloco-jax`) in a single repository.

**Advantages:**
- Completely independent versioning
- Installing one framework can never accidentally trigger the other
- Common pattern in large projects (e.g., `jax` / `jaxlib`)

**Trade-offs:**
- More complex release process (two packages to version and publish)
- Harder to share utilities or documentation
- Users need to know which package to install
- Overkill for a small library with a single algorithm

**Verdict:** Better suited for larger projects with many contributors. For a focused optimizer library, a single package with optional dependencies is simpler.

---

## Alternative 3: Pure Optax Implementation Only

Ship only the JAX/Optax version. The `muloco_wrapper` is already framework-agnostic within JAX (wraps any `GradientTransformation`), and optax is the standard optimizer library in JAX.

For PyTorch users, recommend using the optax wrapper via `torch2jax` or similar bridges, or provide a minimal standalone PyTorch script without packaging.

**Advantages:**
- Single implementation to maintain
- Optax's functional API is more composable
- Cleaner codebase

**Trade-offs:**
- Excludes the large PyTorch user base
- PyTorch-JAX bridges add friction and are not production-ready
- The PyTorch implementation is already written and tested

**Verdict:** Would limit adoption. Both implementations are small enough that maintaining them is low-effort.

---

## Alternative 4: Contrib to Existing Libraries

Instead of a standalone package, contribute the MuLoCo wrapper directly to:
- **optax** as `optax.contrib.muloco_wrapper`
- **torchtitan** or **composer** for PyTorch distributed training

**Advantages:**
- Wider visibility and adoption
- Maintained by the library's community
- No separate package to install

**Trade-offs:**
- Subject to the library's review process and release schedule
- Less control over the API and implementation
- May need to conform to different coding standards
- The outer optimizer wrapper is generic enough to live standalone

**Verdict:** Worth pursuing as a long-term goal (especially for optax), but a standalone package provides immediate availability and faster iteration. The optax contribution could happen in parallel.

---

## Alternative 5: Configuration-Driven Approach

```python
from muloco import build_optimizer

config = {
    "framework": "pytorch",
    "inner": {"type": "muon", "lr": 0.02},
    "outer": {"type": "nesterov_sgd", "lr": 0.7, "momentum": 0.6},
    "sync_interval": 30,
}
optimizer = build_optimizer(config, model.parameters())
```

A configuration-driven factory that builds optimizers from dictionaries or YAML files.

**Advantages:**
- Easy integration with experiment tracking (wandb, hydra)
- Serializable configurations
- Can validate configs before training starts

**Trade-offs:**
- Adds an abstraction layer that obscures what's happening
- Users lose IDE autocompletion and type checking
- Configuration validation logic adds complexity
- The optimizer already has few enough parameters that direct construction is clear

**Verdict:** Adds complexity without proportional benefit for a library with a small, well-defined API. Users who want config-driven construction can wrap the existing API with hydra or similar tools.

---

## Recommendation

The current approach (separate subpackages with optional dependencies) strikes the best balance:
- **Simple:** Direct imports, no abstraction layers, no configuration DSL
- **Flexible:** Install only what you need
- **Explicit:** Clear import paths tell you which framework you're using
- **Maintainable:** Small codebase, each file is self-contained
