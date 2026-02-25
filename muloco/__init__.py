"""MuLoCo: Muon is all you need for Distributed Optimization.

Reference: Therien et al., "MuLoCo: Muon is all you need for
Distributed Optimization", 2025.

Subpackages:
    muloco.pytorch — PyTorch implementation (MuLoCo1, Muon)
    muloco.jax     — JAX/Optax implementation (muloco, muloco_wrapper, diloco)
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("muloco")
except Exception:
    __version__ = "0.1.0.dev0"

# Convenience re-exports when PyTorch is available
try:
    from muloco.pytorch import MuLoCo1, Muon  # noqa: F401
except ImportError:
    pass
