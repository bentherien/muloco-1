# MuLoCo-1: Single-worker MuLoCo optimizer in JAX/Optax.
#
# Implements MuLoCo K=1 (Algorithm 1 from the MuLoCo paper) as an
# optax GradientTransformation wrapper. The inner optimizer (e.g., Muon)
# runs for H steps, then an outer Nesterov SGD step is applied using
# the pseudogradient (parameter delta).
#
# Reference: Therien et al., "MuLoCo: Muon is all you need for
# Distributed Optimization", 2025.

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base


class MuLoCoState(NamedTuple):
    """State for the MuLoCo-1 optimizer.

    Attributes:
        inner_state: State of the inner optimizer (e.g., Muon).
        inner_count: Number of inner steps since the last outer step.
        param_snapshot: Parameter snapshot from the last outer step.
        outer_momentum_buffer: Nesterov momentum buffer for the outer optimizer.
    """
    inner_state: Any
    inner_count: chex.Array  # shape=(), dtype=jnp.int32
    param_snapshot: base.Params
    outer_momentum_buffer: base.Updates


def muloco_wrapper(
    inner_optimizer: base.GradientTransformation,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.6,
    sync_interval: int = 30,
) -> base.GradientTransformation:
    """Wrap any inner optimizer with MuLoCo/DiLoCo K=1 outer Nesterov SGD.

    Every ``sync_interval`` inner steps, computes the pseudogradient
    (parameter delta) and applies an outer Nesterov SGD update.

    Algorithm (K=1, from MuLoCo paper Algorithm 1):
        For each sync round n = 1..N:
            1. Save parameter snapshot: theta_ref = theta
            2. Run H inner optimizer steps
            3. Compute pseudogradient: delta = theta_ref - theta
            4. Update outer momentum: u = mu * u + eta_out * delta
            5. Nesterov update: theta = theta_ref - mu * u - eta_out * delta
            6. Update snapshot: theta_ref = theta

    Args:
        inner_optimizer: The inner optax GradientTransformation
            (e.g., ``optax.contrib.muon`` for MuLoCo, ``optax.adamw`` for DiLoCo).
        outer_lr: Learning rate for the outer Nesterov SGD (eta_out).
        outer_momentum: Momentum coefficient for the outer Nesterov SGD (mu).
        sync_interval: Number of inner steps between outer updates (H).

    Returns:
        An ``optax.GradientTransformation`` implementing MuLoCo/DiLoCo K=1.
    """
    if sync_interval < 1:
        raise ValueError(f"sync_interval must be >= 1, got {sync_interval}")

    def init_fn(params):
        inner_state = inner_optimizer.init(params)
        return MuLoCoState(
            inner_state=inner_state,
            inner_count=jnp.zeros([], jnp.int32),
            param_snapshot=jax.tree.map(jnp.array, params),
            outer_momentum_buffer=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "MuLoCo requires params to be passed to update(). "
                "Use opt.update(grads, state, params) instead of "
                "opt.update(grads, state)."
            )

        # Always run the inner optimizer to keep its state up to date.
        inner_updates, new_inner_state = inner_optimizer.update(
            updates, state.inner_state, params
        )

        new_inner_count = state.inner_count + 1
        is_outer_step = new_inner_count >= sync_interval

        # --- Compute outer step quantities ---
        # These are always computed but only applied when is_outer_step is True.
        # This ensures JIT-compatibility (no Python-level control flow on values).

        # What params would be after applying the inner update:
        theta_after_inner = jax.tree.map(jnp.add, params, inner_updates)

        # Pseudogradient: delta = snapshot - theta_after_inner
        delta = jax.tree.map(
            jnp.subtract, state.param_snapshot, theta_after_inner
        )

        # Outer momentum: u_new = mu * u_old + eta_out * delta
        new_outer_mom = jax.tree.map(
            lambda u, d: outer_momentum * u + outer_lr * d,
            state.outer_momentum_buffer, delta,
        )

        # Outer Nesterov: theta_new = snapshot - mu * u_new - eta_out * delta
        theta_new = jax.tree.map(
            lambda s, u, d: s - outer_momentum * u - outer_lr * d,
            state.param_snapshot, new_outer_mom, delta,
        )

        # Express outer update relative to current params:
        # outer_updates = theta_new - params (so params + updates = theta_new)
        outer_updates = jax.tree.map(jnp.subtract, theta_new, params)

        # --- Select between inner and outer updates ---
        final_updates = jax.tree.map(
            lambda iu, ou: jnp.where(is_outer_step, ou, iu),
            inner_updates, outer_updates,
        )

        # --- Conditionally update state ---
        new_snapshot = jax.tree.map(
            lambda s, t: jnp.where(is_outer_step, t, s),
            state.param_snapshot, theta_new,
        )
        final_outer_mom = jax.tree.map(
            lambda old, new: jnp.where(is_outer_step, new, old),
            state.outer_momentum_buffer, new_outer_mom,
        )
        final_inner_count = jnp.where(
            is_outer_step, jnp.int32(0), new_inner_count
        )

        new_state = MuLoCoState(
            inner_state=new_inner_state,
            inner_count=final_inner_count,
            param_snapshot=new_snapshot,
            outer_momentum_buffer=final_outer_mom,
        )

        return final_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def muloco(
    learning_rate: base.ScalarOrSchedule,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.6,
    sync_interval: int = 30,
    ns_coeffs: Union[
        Tuple[float, float, float],
        Tuple[Tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    muon_weight_dimension_numbers=None,
) -> base.GradientTransformation:
    """MuLoCo K=1: Muon inner optimizer + Nesterov SGD outer optimizer.

    Combines ``optax.contrib.muon`` as the inner optimizer with an outer
    Nesterov SGD that fires every ``sync_interval`` steps, using the
    pseudogradient (parameter delta) to update parameters.

    For K=1, this is equivalent to a single-worker variant of MuLoCo
    (Algorithm 1 from the paper). The inner Muon optimizer applies
    Newton-Schulz orthogonalization to 2D weight matrices and Adam to
    non-2D parameters (embeddings, norms, biases).

    Args:
        learning_rate: Inner learning rate for Muon (can be a schedule).
        outer_lr: Outer Nesterov SGD learning rate (eta_out).
        outer_momentum: Outer Nesterov SGD momentum (mu).
        sync_interval: Inner steps between outer updates (H).
        ns_coeffs: Newton-Schulz iteration coefficients.
        ns_steps: Number of Newton-Schulz iterations (ignored if ns_coeffs
            is a tuple of tuples).
        beta: Momentum decay rate for inner Muon.
        eps: Numerical stability epsilon.
        weight_decay: Weight decay for Muon parameters.
        weight_decay_mask: Mask for weight decay.
        mu_dtype: Data type for momentum accumulator.
        nesterov: Whether to use Nesterov momentum in the inner optimizer.
        adaptive: Whether to use adaptive scaling in Muon.
        adam_b1: Adam beta1 for non-Muon parameters.
        adam_b2: Adam beta2 for non-Muon parameters.
        adam_eps_root: Adam epsilon root for non-Muon parameters.
        adam_weight_decay: Adam weight decay for non-Muon parameters.
        muon_weight_dimension_numbers: Specification for which parameters
            use Muon vs Adam. See ``optax.contrib.muon`` for details.

    Returns:
        An ``optax.GradientTransformation`` implementing MuLoCo K=1.
    """
    inner = optax.contrib.muon(
        learning_rate=learning_rate,
        ns_coeffs=ns_coeffs,
        ns_steps=ns_steps,
        beta=beta,
        eps=eps,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
        adaptive=adaptive,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
        adam_weight_decay=adam_weight_decay,
        muon_weight_dimension_numbers=muon_weight_dimension_numbers,
    )
    return muloco_wrapper(inner, outer_lr, outer_momentum, sync_interval)


def diloco(
    learning_rate: base.ScalarOrSchedule,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    sync_interval: int = 30,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """DiLoCo K=1: AdamW inner optimizer + Nesterov SGD outer optimizer.

    Convenience function that uses ``optax.adamw`` as the inner optimizer
    with the ``muloco_wrapper`` outer Nesterov SGD.

    Args:
        learning_rate: Inner learning rate for AdamW (can be a schedule).
        outer_lr: Outer Nesterov SGD learning rate (eta_out).
        outer_momentum: Outer Nesterov SGD momentum (mu).
        sync_interval: Inner steps between outer updates (H).
        b1: AdamW beta1.
        b2: AdamW beta2.
        eps: Numerical stability epsilon.
        eps_root: AdamW epsilon root.
        weight_decay: AdamW weight decay.
        mu_dtype: Data type for momentum accumulator.
        nesterov: Whether to use Nesterov momentum in AdamW.

    Returns:
        An ``optax.GradientTransformation`` implementing DiLoCo K=1.
    """
    inner = optax.adamw(
        learning_rate=learning_rate,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        weight_decay=weight_decay,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
    )
    return muloco_wrapper(inner, outer_lr, outer_momentum, sync_interval)
