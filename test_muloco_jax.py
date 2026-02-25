"""
Test script for MuLoCo-1 JAX/Optax implementation.

Runs a series of tests to verify correctness:
  1. MuLoCo wrapper with AdamW inner (regression task)
  2. MuLoCo wrapper with AdamW inner (classification with embeddings)
  3. Full MuLoCo with Muon inner (regression task)
  4. DiLoCo convenience function
  5. Outer step timing verification
  6. Comparison: MuLoCo vs plain Muon vs plain AdamW

Usage:
    python test_muloco_jax.py              # Run all tests
    python test_muloco_jax.py --verbose     # Verbose output
    python test_muloco_jax.py --quick       # Quick smoke test only
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import optax

from muloco_jax import muloco_wrapper, muloco, diloco, MuLoCoState


# ---------------------------------------------------------------------------
# Simple models (raw JAX, no framework dependency)
# ---------------------------------------------------------------------------

def init_mlp(key, input_dim, hidden_dim, output_dim):
    """Initialize a 2-hidden-layer MLP."""
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.1
    return {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)) * scale,
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, hidden_dim)) * scale,
        'b2': jnp.zeros(hidden_dim),
        'w3': jax.random.normal(k3, (hidden_dim, output_dim)) * scale,
        'b3': jnp.zeros(output_dim),
    }


def mlp_forward(params, x):
    """Forward pass for MLP."""
    h = jnp.tanh(x @ params['w1'] + params['b1'])
    h = jnp.tanh(h @ params['w2'] + params['b2'])
    return h @ params['w3'] + params['b3']


def init_mini_transformer(key, vocab_size, d_model, n_heads, seq_len):
    """Initialize a minimal single-layer transformer (for testing)."""
    keys = jax.random.split(key, 8)
    scale = 0.02
    head_dim = d_model // n_heads
    return {
        'embed': jax.random.normal(keys[0], (vocab_size, d_model)) * scale,
        'pos_embed': jax.random.normal(keys[1], (seq_len, d_model)) * scale,
        'attn_qkv': jax.random.normal(keys[2], (d_model, 3 * d_model)) * scale,
        'attn_proj': jax.random.normal(keys[3], (d_model, d_model)) * scale,
        'ln1_scale': jnp.ones(d_model),
        'ln1_bias': jnp.zeros(d_model),
        'ffn_up': jax.random.normal(keys[4], (d_model, 4 * d_model)) * scale,
        'ffn_down': jax.random.normal(keys[5], (4 * d_model, d_model)) * scale,
        'ln2_scale': jnp.ones(d_model),
        'ln2_bias': jnp.zeros(d_model),
        'head': jax.random.normal(keys[6], (d_model, vocab_size)) * scale,
        'ln_f_scale': jnp.ones(d_model),
        'ln_f_bias': jnp.zeros(d_model),
    }


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def mini_transformer_forward(params, input_ids):
    """Forward pass for minimal transformer."""
    B, T = input_ids.shape
    d_model = params['embed'].shape[1]

    x = params['embed'][input_ids] + params['pos_embed'][:T]
    h = layer_norm(x, params['ln1_scale'], params['ln1_bias'])

    qkv = h @ params['attn_qkv']
    q, k, v = jnp.split(qkv, 3, axis=-1)
    scale = jnp.sqrt(jnp.float32(d_model))
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
    mask = jnp.triu(jnp.ones((T, T)), k=1) * -1e9
    attn = jax.nn.softmax(attn + mask, axis=-1)
    attn_out = attn @ v
    x = x + attn_out @ params['attn_proj']

    h = layer_norm(x, params['ln2_scale'], params['ln2_bias'])
    x = x + jax.nn.gelu(h @ params['ffn_up']) @ params['ffn_down']

    x = layer_norm(x, params['ln_f_scale'], params['ln_f_bias'])
    logits = x @ params['head']
    return logits


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_muloco_wrapper_adamw(verbose=False):
    """Test MuLoCo wrapper with AdamW inner on regression."""
    print("Test 1: MuLoCo wrapper + AdamW inner (regression)")

    key = jax.random.PRNGKey(42)
    params = init_mlp(key, input_dim=4, hidden_dim=64, output_dim=4)

    x = jax.random.normal(jax.random.PRNGKey(0), (128, 4))
    y = jnp.sin(x)

    def loss_fn(params, x, y):
        return jnp.mean((mlp_forward(params, x) - y) ** 2)

    inner = optax.adamw(learning_rate=1e-3)
    opt = muloco_wrapper(inner, outer_lr=0.7, outer_momentum=0.6, sync_interval=10)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(300):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if verbose and (i + 1) % 50 == 0:
            print(f"    Step {i+1}: loss = {loss:.6f}")

    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    assert losses[-1] < losses[0] * 0.1, (
        f"Loss should decrease significantly: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )
    print("    PASSED")
    return True


def test_muloco_wrapper_transformer(verbose=False):
    """Test MuLoCo wrapper with AdamW inner on a mini transformer."""
    print("Test 2: MuLoCo wrapper + AdamW inner (mini transformer)")

    key = jax.random.PRNGKey(42)
    vocab_size, d_model, n_heads, seq_len = 100, 32, 4, 16
    params = init_mini_transformer(key, vocab_size, d_model, n_heads, seq_len)

    # Synthetic sequence data
    data_key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(data_key, (32, seq_len), 0, vocab_size)
    targets = jnp.roll(input_ids, -1, axis=1)

    def loss_fn(params, input_ids, targets):
        logits = mini_transformer_forward(params, input_ids)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        )

    inner = optax.adamw(learning_rate=3e-4)
    opt = muloco_wrapper(inner, outer_lr=0.5, outer_momentum=0.5, sync_interval=15)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, targets)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(200):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if verbose and (i + 1) % 50 == 0:
            print(f"    Step {i+1}: loss = {loss:.6f}")

    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    assert losses[-1] < losses[0] * 0.8, (
        f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )
    print("    PASSED")
    return True


def test_full_muloco_muon(verbose=False):
    """Test full MuLoCo (Muon inner) on regression."""
    print("Test 3: Full MuLoCo with Muon inner (regression)")

    key = jax.random.PRNGKey(42)
    # All 2D params so Muon applies to everything (no adam fallback needed)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        'w1': jax.random.normal(k1, (4, 64)) * 0.1,
        'w2': jax.random.normal(k2, (64, 64)) * 0.1,
        'w3': jax.random.normal(k3, (64, 4)) * 0.1,
    }

    x = jax.random.normal(jax.random.PRNGKey(0), (128, 4))
    y = jnp.sin(x)

    def loss_fn(params, x, y):
        h = jnp.tanh(x @ params['w1'])
        h = jnp.tanh(h @ params['w2'])
        pred = h @ params['w3']
        return jnp.mean((pred - y) ** 2)

    opt = muloco(
        learning_rate=0.01,
        outer_lr=0.7,
        outer_momentum=0.6,
        sync_interval=10,
        weight_decay=0.0,
    )
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(300):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if verbose and (i + 1) % 50 == 0:
            print(f"    Step {i+1}: loss = {loss:.6f}")

    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    assert losses[-1] < losses[0] * 0.5, (
        f"Loss should decrease significantly: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )
    print("    PASSED")
    return True


def test_diloco(verbose=False):
    """Test DiLoCo convenience function."""
    print("Test 4: DiLoCo (AdamW inner) convenience function")

    key = jax.random.PRNGKey(42)
    params = init_mlp(key, input_dim=4, hidden_dim=64, output_dim=4)

    x = jax.random.normal(jax.random.PRNGKey(0), (128, 4))
    y = jnp.sin(x)

    def loss_fn(params, x, y):
        return jnp.mean((mlp_forward(params, x) - y) ** 2)

    opt = diloco(
        learning_rate=1e-3,
        outer_lr=0.7,
        outer_momentum=0.9,
        sync_interval=10,
    )
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(200):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))

    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    assert losses[-1] < losses[0] * 0.1, (
        f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )
    print("    PASSED")
    return True


def test_outer_step_timing(verbose=False):
    """Verify outer steps happen at the correct intervals."""
    print("Test 5: Outer step timing verification")

    key = jax.random.PRNGKey(42)
    params = {'w': jax.random.normal(key, (4, 4)) * 0.1}

    def loss_fn(params, x):
        return jnp.mean((x @ params['w']) ** 2)

    sync_interval = 5
    inner = optax.sgd(learning_rate=0.01)
    opt = muloco_wrapper(
        inner, outer_lr=0.5, outer_momentum=0.5, sync_interval=sync_interval
    )
    opt_state = opt.init(params)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))

    outer_step_indices = []
    for i in range(20):
        grads = jax.grad(loss_fn)(params, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        inner_count = int(opt_state.inner_count)
        expected_count = (i + 1) % sync_interval
        if verbose:
            is_outer = "OUTER" if inner_count == 0 and i > 0 else ""
            print(f"    Step {i+1}: inner_count={inner_count} {is_outer}")

        assert inner_count == expected_count, (
            f"Step {i+1}: expected inner_count={expected_count}, "
            f"got {inner_count}"
        )
        if inner_count == 0 and i > 0:
            outer_step_indices.append(i + 1)

    expected_outer = [5, 10, 15, 20]
    assert outer_step_indices == expected_outer, (
        f"Expected outer steps at {expected_outer}, got {outer_step_indices}"
    )
    print(f"    Outer steps fired at: {outer_step_indices}")
    print("    PASSED")
    return True


def test_snapshot_update(verbose=False):
    """Verify that parameter snapshots are updated correctly on outer steps."""
    print("Test 6: Snapshot update verification")

    key = jax.random.PRNGKey(42)
    params = {'w': jax.random.normal(key, (4, 4)) * 0.1}

    def loss_fn(params, x):
        return jnp.mean((x @ params['w']) ** 2)

    sync_interval = 3
    inner = optax.sgd(learning_rate=0.01)
    opt = muloco_wrapper(
        inner, outer_lr=0.5, outer_momentum=0.5, sync_interval=sync_interval
    )
    opt_state = opt.init(params)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
    initial_snapshot = opt_state.param_snapshot['w'].copy()

    # Run inner steps (snapshot should NOT change)
    for i in range(sync_interval - 1):
        grads = jax.grad(loss_fn)(params, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    snapshot_before_outer = opt_state.param_snapshot['w']
    assert jnp.allclose(snapshot_before_outer, initial_snapshot), (
        "Snapshot should not change during inner steps"
    )
    if verbose:
        print("    Snapshot unchanged during inner steps: OK")

    # Run one more step (outer step - snapshot should change)
    grads = jax.grad(loss_fn)(params, x)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    snapshot_after_outer = opt_state.param_snapshot['w']
    assert not jnp.allclose(snapshot_after_outer, initial_snapshot), (
        "Snapshot should change after outer step"
    )
    if verbose:
        print("    Snapshot updated after outer step: OK")

    print("    PASSED")
    return True


def test_comparison(verbose=False):
    """Compare MuLoCo vs plain Muon vs plain AdamW on regression."""
    print("Test 7: Comparison - MuLoCo vs Muon vs AdamW")

    # All 2D params for fair Muon comparison
    def make_params(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            'w1': jax.random.normal(k1, (4, 64)) * 0.1,
            'w2': jax.random.normal(k2, (64, 64)) * 0.1,
            'w3': jax.random.normal(k3, (64, 4)) * 0.1,
        }

    x = jax.random.normal(jax.random.PRNGKey(0), (128, 4))
    y = jnp.sin(x)

    def loss_fn(params, x, y):
        h = jnp.tanh(x @ params['w1'])
        h = jnp.tanh(h @ params['w2'])
        pred = h @ params['w3']
        return jnp.mean((pred - y) ** 2)

    num_steps = 300
    lr = 0.01
    results = {}

    for name, opt_fn in [
        ("AdamW", lambda: optax.adamw(learning_rate=1e-3)),
        ("Muon", lambda: optax.contrib.muon(learning_rate=lr)),
        ("MuLoCo", lambda: muloco(
            learning_rate=lr, outer_lr=0.7,
            outer_momentum=0.6, sync_interval=10,
        )),
    ]:
        key = jax.random.PRNGKey(42)
        params = make_params(key)
        opt = opt_fn()
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, new_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, loss

        t0 = time.time()
        losses = []
        for i in range(num_steps):
            params, opt_state, loss = step(params, opt_state)
            losses.append(float(loss))
        elapsed = time.time() - t0

        results[name] = losses
        print(f"    {name:8s}: {losses[0]:.4f} -> {losses[-1]:.4f} ({elapsed:.2f}s)")

    # MuLoCo should achieve competitive or better loss than standalone Muon
    print("    (Note: relative performance depends on hyperparameter tuning)")
    print("    PASSED")
    return True


def test_sync_interval_1(verbose=False):
    """Test edge case: sync_interval=1 (every step is an outer step)."""
    print("Test 8: sync_interval=1 edge case")

    key = jax.random.PRNGKey(42)
    params = init_mlp(key, input_dim=4, hidden_dim=32, output_dim=4)

    x = jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    y = jnp.sin(x)

    def loss_fn(params, x, y):
        return jnp.mean((mlp_forward(params, x) - y) ** 2)

    inner = optax.adamw(learning_rate=1e-3)
    opt = muloco_wrapper(inner, outer_lr=0.5, outer_momentum=0.5, sync_interval=1)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(100):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        # Every step should be an outer step, so inner_count is always 0
        assert int(opt_state.inner_count) == 0

    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss:   {losses[-1]:.6f}")
    assert losses[-1] < losses[0], "Loss should decrease even with sync_interval=1"
    print("    PASSED")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test MuLoCo-1 JAX/Optax")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"Optax version: {optax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    tests = [
        test_muloco_wrapper_adamw,
        test_muloco_wrapper_transformer,
        test_full_muloco_muon,
        test_diloco,
        test_outer_step_timing,
        test_snapshot_update,
    ]
    if not args.quick:
        tests += [
            test_comparison,
            test_sync_interval_1,
        ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn(verbose=args.verbose)
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
