import gpt2 as nn
import jax.numpy as jnp
import jax
import optax
import optax.losses as ll
from flax.training import train_state
import dataloader as dl
import time

# Create a train state
def create_train_state(model, init_key, learning_rate):
  dummy = jnp.ones((1,1), dtype=int)
  params = model.init(init_key, dummy)
  optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)

  return train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=optimizer
  )

# Create train step
@jax.jit
def train_step(state, x, y):
  def loss_f(params):
    logits = state.apply_fn(params, x)
    return ll.softmax_cross_entropy_with_integer_labels(logits, y).mean()

  loss, grads = jax.value_and_grad(loss_f)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


if __name__ == "__main__":
    # TODO
    # - Gradient norm clipping
    # - Learning rate scheduler (cosine: max_lr: 6e-4, min_lr = max_lr*0.1, warmup_steps = 10, max_steps = 50)
    # - gradient accumulation so that we match the effective batch size used by GPT2 of 0.5M tokens
    # - JAXify the loops

    # Get model definition
    config = nn.Config(vocab_size=50304)
    model = nn.GPT2(config)

    # Initialise model and training state
    key = jax.random.PRNGKey(42069)
    key, init_key = jax.random.split(key)
    state = create_train_state(model, init_key, learning_rate=3e-4)

    # Training LOOP
    loader = dl.DataLoaderLite(B=16, T=1024, fname='../data/input.txt')
    for i in range(50):
        t0 = time.time()
        x,y = loader.next_batch()
        state, loss = train_step(state, x, y)
        jax.block_until_ready(state)
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_per_sec = (loader.B * loader.T) / (t1 - t0)
        print(f"Step: {i}, Loss: {loss}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec}")