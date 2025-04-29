import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import optax
import optax.losses as ll
from flax.training import train_state

import gpt2 as nn
import dataloader as dl

# -----------------------------------------------------------------------------
# 1) Create the TrainState
# -----------------------------------------------------------------------------
def create_train_state(model, init_key, learning_rate):
    dummy = jnp.ones((1,1), dtype=jnp.int32)
    params = model.init(init_key, dummy)
    tx = optax.adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

# -----------------------------------------------------------------------------
# 2) A single-step JIT-ed train function
# -----------------------------------------------------------------------------
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)                     # (B, T, V)
        return ll.softmax_cross_entropy_with_integer_labels(
            logits, y
        ).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# -----------------------------------------------------------------------------
# 3) Bundle N steps into one jax.lax.scan
# -----------------------------------------------------------------------------
def make_multi_step(n_steps):
    """Returns a jitted function that does `n_steps` updates in one go."""
    @jax.jit
    def multi_step(state, xs, ys):
        # xs, ys: arrays of shape (n_steps, B, T)
        def body(carry, batch):
            st, _ = carry
            x, y = batch
            st, loss = train_step(st, x, y)
            return (st, loss), loss

        # Initialize carry=(state, dummy_loss) and scan over the batches
        (final_state, _), losses = lax.scan(
            body,
            (state, 0.0),
            (xs, ys)
        )
        return final_state, losses   # losses.shape == (n_steps,)
    return multi_step

# -----------------------------------------------------------------------------
# 4) Main script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparams
    B = 16
    T = 1024
    learning_rate = 3e-4
    total_steps = 50
    n_steps = 10

    # Model + State init
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    config = nn.Config(vocab_size=50304)
    model  = nn.GPT2(config)
    state  = create_train_state(model, init_key, learning_rate)

    # Data
    loader = dl.DataLoaderLite(B=B, T=T, fname='../data/input.txt')

    # Compile & warm up:
    #  - one train_step
    x0, y0 = loader.next_batch()
    state, _ = train_step(state, x0, y0)
    jax.block_until_ready((state, _))

    #  - one multi_step
    multi_step = make_multi_step(n_steps)
    # build dummy xs,ys of shape (n_steps, B, T)
    xs0 = jnp.stack([loader.next_batch()[0] for _ in range(n_steps)])
    ys0 = jnp.stack([loader.next_batch()[1] for _ in range(n_steps)])
    state, _ = multi_step(state, xs0, ys0)
    jax.block_until_ready((state, _))

    # Timed loop
    calls = total_steps // n_steps
    print(f"Running {total_steps} steps as {calls} calls of {n_steps}…")
    for call_i in range(calls):
        # grab next n_steps batches
        xs = jnp.stack([loader.next_batch()[0] for _ in range(n_steps)])
        ys = jnp.stack([loader.next_batch()[1] for _ in range(n_steps)])

        t0 = time.time()
        state, losses = multi_step(state, xs, ys)
        # wait for the device work to finish for accurate timing
        jax.block_until_ready((state, losses))
        t1 = time.time()

        dt = (t1 - t0)*1000
        avg_loss = losses.mean()
        tok_per_sec = (B * T * n_steps) / (t1 - t0)

        print(f"Call {call_i+1}/{calls}: dt={dt:.1f}ms, "
              f"avg loss={avg_loss:.4f}, "
              f"≈{tok_per_sec:.1f} tok/s")
