"""MuLoCo PyTorch implementation.

Provides the MuLoCo-1 optimizer (Muon inner + Nesterov SGD outer)
and the standalone Muon optimizer for single-GPU use.

Example::

    from muloco.pytorch import MuLoCo1, Muon

    # MuLoCo-1 with inner Muon + outer Nesterov SGD
    optimizer = MuLoCo1(param_groups, inner_lr=0.02, outer_lr=0.7)

    # Standalone Muon
    optimizer = Muon(param_groups, lr=0.02)
"""

try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "muloco.pytorch requires PyTorch. "
        "Install it with: pip install muloco[pytorch]"
    ) from e

from muloco.pytorch.muloco import (  # noqa: F401
    MuLoCo1,
    Muon,
    adjust_lr_spectral_norm,
    zeropower_via_newtonschulz5,
)
