"""MuLoCo JAX/Optax implementation.

Provides the MuLoCo optimizer as an optax GradientTransformation,
including convenience functions for MuLoCo (Muon inner) and DiLoCo
(AdamW inner).

Example::

    from muloco.jax import muloco, muloco_wrapper, diloco

    # MuLoCo with Muon inner optimizer
    opt = muloco(learning_rate=0.02, outer_lr=0.7, sync_interval=30)

    # DiLoCo with AdamW inner optimizer
    opt = diloco(learning_rate=1e-3, outer_lr=0.7, sync_interval=30)

    # Generic wrapper around any optax optimizer
    inner = optax.adamw(learning_rate=1e-3)
    opt = muloco_wrapper(inner, outer_lr=0.7, sync_interval=30)
"""

try:
    import jax  # noqa: F401
    import optax  # noqa: F401
except ImportError as e:
    raise ImportError(
        "muloco.jax requires JAX and Optax. "
        "Install them with: pip install muloco[jax]"
    ) from e

from muloco.jax.muloco import (  # noqa: F401
    MuLoCoState,
    diloco,
    muloco,
    muloco_wrapper,
)
