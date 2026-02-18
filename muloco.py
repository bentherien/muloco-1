# MuLoCo-1: Single-worker MuLoCo optimizer.
# Muon inner optimizer + SGD with Nesterov momentum outer optimizer.
#
# Reference: MuLoCo paper, Algorithm 1 with K=1.
#
# The inner Muon optimizer implementation is taken directly from
# dion_optimizer/muon.py (zeropower_via_newtonschulz5, momentum update,
# spectral norm LR scaling) and adapted for standalone single-GPU use.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (from muon.py)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to approximate the orthogonalization of G.
    Produces the nearest orthogonal matrix to G via 5 iterations.
    """
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adjust_lr_spectral_norm(lr: float, param_shape) -> float:
    """Adjust LR from spectral norm 1 to RMS operator norm 1."""
    fan_out, fan_in = param_shape[:2]
    return lr * math.sqrt(fan_out / fan_in)


# ---------------------------------------------------------------------------
# Inner Muon optimizer (single-GPU, adapted from muon.py)
# ---------------------------------------------------------------------------

class _InnerMuon(Optimizer):
    """
    Single-GPU Muon optimizer for use as the inner optimizer in MuLoCo-1.

    Implements the same algorithm as dion_optimizer/muon.py:
    - Newton-Schulz orthogonalization for 2D+ matrix parameters
    - AdamW for scalar/embedding/head parameters
    - Spectral norm LR scaling for Muon parameters
    - Momentum with optional Nesterov

    This is a simplified version that removes distributed communication
    (all-to-all, all-gather) since K=1 only needs single-GPU operations.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]

            if algo == "muon":
                self._muon_step(group)
            elif algo == "adamw":
                self._adamw_step(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        return loss

    def _muon_step(self, group: dict):
        """Muon update: momentum + Newton-Schulz orthogonalization."""
        lr = group["lr"]
        mu = group["mu"]
        wd = group["weight_decay"]
        eps = group["epsilon"]
        nesterov = group["nesterov"]
        adjust_lr_mode = group["adjust_lr"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p.data)

            g = p.grad.to(dtype=state["momentum"].dtype)
            m = state["momentum"]

            # Update momentum
            m.mul_(mu).add_(g)

            # Compute input to orthogonalization
            if nesterov:
                u = m * mu + g
            else:
                u = m

            u = u.to(dtype=torch.bfloat16)

            # Newton-Schulz orthogonalization
            original_shape = u.shape
            if u.ndim >= 4:
                u = u.flatten(end_dim=-3)
            ortho = zeropower_via_newtonschulz5(u, epsilon=eps)
            ortho = ortho.reshape(original_shape)

            # Compute adjusted learning rate
            if adjust_lr_mode == "spectral_norm":
                adj_lr = adjust_lr_spectral_norm(lr, p.shape)
            elif adjust_lr_mode == "rms_norm":
                A, B = p.shape[:2]
                adj_lr = lr * 0.2 * math.sqrt(max(A, B))
            else:
                adj_lr = lr

            # Weight decay
            p.data.mul_(1 - lr * wd)

            # Weight update
            p.data.add_(ortho.to(dtype=p.data.dtype), alpha=-adj_lr)

    def _adamw_step(self, group: dict):
        """Standard AdamW update for scalar/embedding/head parameters."""
        lr = group["lr"]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        wd = group["weight_decay"]
        eps = group["epsilon"]
        step = group["step"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p.data)
                state["variance"] = torch.zeros_like(p.data)

            g = p.grad
            m = state["momentum"]
            v = state["variance"]

            # Update moments
            m.mul_(beta1).add_(g, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Bias correction
            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step
            m_hat = m / bc1
            v_hat = v / bc2

            # Weight decay
            p.data.mul_(1 - lr * wd)

            # Update
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


# ---------------------------------------------------------------------------
# MuLoCo-1 optimizer
# ---------------------------------------------------------------------------

class MuLoCo1:
    """
    MuLoCo K=1: Single-worker MuLoCo optimizer.

    Wraps the Muon optimizer as an inner optimizer and applies SGD with
    Nesterov momentum as the outer optimizer every `sync_interval` steps.

    Algorithm (from the MuLoCo paper, Algorithm 1 with K=1):
        For each sync round:
            1. Save parameter snapshot: theta_ref = theta.clone()
            2. Run H inner Muon steps
            3. Compute pseudogradient: delta = theta_ref - theta
            4. Update outer momentum: u = mu * u_old + eta_out * delta
            5. Apply Nesterov update: theta = theta_ref - mu * u - eta_out * delta

    Args:
        params: Model parameters or parameter groups for the inner Muon optimizer.
            Parameter groups should have an 'algorithm' key: 'muon' for 2D+
            matrix params, 'adamw' for scalar/embedding/head params.
        inner_lr: Learning rate for the inner Muon optimizer.
        outer_lr: Learning rate for the outer SGD optimizer.
        outer_momentum: Momentum coefficient for the outer Nesterov SGD.
        sync_interval: Number of inner steps between outer updates (H).
        mu: Momentum factor for inner Muon.
        betas: Betas for inner AdamW (used for non-matrix parameters).
        weight_decay: Weight decay for the inner optimizer.
        epsilon: Epsilon for numerical stability.
        nesterov: Whether to use Nesterov momentum in the inner Muon.
        adjust_lr: LR adjustment for Muon ("spectral_norm", "rms_norm", None).
    """

    def __init__(
        self,
        params,
        inner_lr: float = 0.02,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.6,
        sync_interval: int = 30,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
    ):
        if outer_lr < 0.0:
            raise ValueError(f"Invalid outer learning rate: {outer_lr}")
        if not 0.0 <= outer_momentum < 1.0:
            raise ValueError(f"Invalid outer momentum: {outer_momentum}")
        if sync_interval < 1:
            raise ValueError(f"Invalid sync interval: {sync_interval}")

        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.sync_interval = sync_interval
        self.inner_step_count = 0
        self.outer_step_count = 0

        # Create the inner Muon optimizer
        self.inner_optimizer = _InnerMuon(
            params=params,
            lr=inner_lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
        )

        # Collect all parameters across all groups
        self._all_params: List[Tensor] = []
        for group in self.inner_optimizer.param_groups:
            for p in group["params"]:
                self._all_params.append(p)

        # Initialize outer optimizer state
        self._param_snapshots: List[Tensor] = [
            p.data.clone() for p in self._all_params
        ]
        self._outer_momentum_buffers: List[Tensor] = [
            torch.zeros_like(p.data) for p in self._all_params
        ]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Runs one inner Muon step. Every `sync_interval` inner steps,
        also applies the outer Nesterov SGD update.
        """
        loss = self.inner_optimizer.step(closure)
        self.inner_step_count += 1

        if self.inner_step_count >= self.sync_interval:
            self._outer_step()
            self.inner_step_count = 0

        return loss

    @torch.no_grad()
    def _outer_step(self):
        """Apply outer Nesterov SGD using the pseudogradient.

        Implements:
            delta = theta_snapshot - theta_current   (pseudogradient)
            u = mu * u_old + eta_out * delta         (momentum update)
            theta = theta_ref - mu * u - eta_out * delta  (Nesterov step)
        """
        self.outer_step_count += 1

        for i, p in enumerate(self._all_params):
            snapshot = self._param_snapshots[i]
            u = self._outer_momentum_buffers[i]

            # Pseudogradient: delta = theta_ref - theta_current
            delta = snapshot - p.data

            # Nesterov SGD outer update
            u.mul_(self.outer_momentum).add_(delta, alpha=self.outer_lr)

            # theta = theta_ref - mu * u - eta_out * delta
            p.data.copy_(snapshot)
            p.data.add_(u, alpha=-self.outer_momentum)
            p.data.add_(delta, alpha=-self.outer_lr)

        # Save new snapshots for the next sync round
        self._param_snapshots = [p.data.clone() for p in self._all_params]

    def zero_grad(self, *args, **kwargs):
        """Zero gradients of all parameters."""
        self.inner_optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        """Access inner optimizer's param groups (for LR scheduling etc.)."""
        return self.inner_optimizer.param_groups

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state as a dictionary."""
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "param_snapshots": [s.clone() for s in self._param_snapshots],
            "outer_momentum_buffers": [
                u.clone() for u in self._outer_momentum_buffers
            ],
            "inner_step_count": self.inner_step_count,
            "outer_step_count": self.outer_step_count,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "sync_interval": self.sync_interval,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from a dictionary."""
        self.inner_optimizer.load_state_dict(state_dict["inner_optimizer"])
        self._param_snapshots = [s.clone() for s in state_dict["param_snapshots"]]
        self._outer_momentum_buffers = [
            u.clone() for u in state_dict["outer_momentum_buffers"]
        ]
        self.inner_step_count = state_dict["inner_step_count"]
        self.outer_step_count = state_dict["outer_step_count"]
        self.outer_lr = state_dict["outer_lr"]
        self.outer_momentum = state_dict["outer_momentum"]
        self.sync_interval = state_dict["sync_interval"]
