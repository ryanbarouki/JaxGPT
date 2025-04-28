import gpt2 as nn
import jax.numpy as jnp
import jax
import optax
import optax.losses as ll
from flax.training import train_state
import dataloader as dl

# Create a train state
def create_train_state(model, init_key, learning_rate):
  dummy = jnp.ones((1,1), dtype=int)
  params = model.init(init_key, dummy)
  optimizer = optax.adamw(learning_rate)

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
    # Get model definition
    config = nn.Config()
    model = nn.GPT2(config)

    # Initialise model and training state
    key = jax.random.PRNGKey(42069)
    key, init_key = jax.random.split(key)
    state = create_train_state(model, init_key, learning_rate=3e-4)

    # Training LOOP
    loader = dl.DataLoaderLite(B=4, T=32, fname='../data/input.txt')
    for i in range(100):
        x,y = loader.next_batch()
        state, loss = train_step(state, x, y)
        print(f"Iter: {i}, loss: {loss}")