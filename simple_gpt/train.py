import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from tqdm import tqdm
import modules as rbnn

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
n_emb = 96
num_heads = 6
num_blocks = 6
learning_rate = 1e-3
num_epochs = 1

def get_batch_pythonic(key, split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generate random starting positions for the examples
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_batch(key, data):
    # how many valid start positions we have
    max_idx = data.shape[0] - (block_size + 1)
    # draw batch_size random starts in [0, max_idx)
    starts = jax.random.randint(key, (batch_size,), 0, max_idx)

    def slice_pair(i):
        # pull [i : i+block_size]  and [i+1 : i+1+block_size]
        x = jax.lax.dynamic_slice(data, (i,), (block_size,))
        y = jax.lax.dynamic_slice(data, (i+1,), (block_size,))
        return x, y

    # vmap over our vector of starts
    xs, ys = jax.vmap(slice_pair)(starts)
    return xs, ys

def inspect_batch():
    key = jax.random.PRNGKey(4206969)
    xb, yb = get_batch(key, 'train')
    print('inputs:')
    print(xb.shape)
    print(xb)
    print('targets:')
    print(yb.shape)
    print(yb)

    print('----')

    for b in range(batch_size): # batch dimension
        for t in range(block_size): # time dimension
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target: {target}")

model = rbnn.GPT(n_emb=n_emb, vocab_size=vocab_size, block_size=block_size, num_heads=num_heads, num_blocks=num_blocks)

def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return optax.losses.softmax_cross_entropy_with_integer_labels(logits, y).mean()

# Initialise Model ===============================
key, init_key = jax.random.split(jax.random.PRNGKey(420696969))
xb, yb = get_batch(key, train_data)
params = model.init(init_key, jnp.ones_like(xb))

#TODO Make this faster
def estimate_loss(key, params, eval_iters=200):
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            key, subkey = jax.random.split(key)
            if split == 'train':
                X, Y = get_batch(subkey, train_data)
            else:
                X, Y = get_batch(subkey, val_data)
            loss = loss_fn(params, X, Y)
            losses.append(float(loss))
        out[split] = sum(losses) / len(losses)
    return out

# Training =======================================
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)

# 1) the single‐step update:
@jax.jit
def _step(carry, rng):
    params, opt_state = carry
    xb, yb = get_batch(rng, train_data)
    loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), loss

# 2) the “run many” scan:
def run_many(params, opt_state, rng, num_steps: int):
    keys = jax.random.split(rng, num_steps + 1)
    (params, opt_state), losses = jax.lax.scan(
        _step,
        (params, opt_state),
        keys[1:],      # the per‐step RNGs
    )
    return params, opt_state, losses, keys[0]

# 3) jit that with num_steps static:
run_many = jax.jit(run_many, static_argnums=(3,))

chunk_size   = 100
num_chunks   = 5000 // chunk_size  # e.g. 5000//100 = 50
run_chunk    = jax.jit(run_many, static_argnums=(3,))  # assume run_many does exactly `chunk_size` steps

pbar = tqdm(range(num_chunks))
for _ in pbar:
    key, subkey = jax.random.split(key)
    params, opt_state, losses, key = run_chunk(params, opt_state, subkey, chunk_size)
    pbar.set_description(f"loss {losses[-1]:.4f}")

# 4) now your outer loop is trivial:
# key = jax.random.PRNGKey(0)
# for epoch in range(num_epochs):
#     key, subkey = jax.random.split(key)
#     params, opt_state, losses, key = run_many(params, opt_state, subkey, num_steps=5000)
#     print(f"Epoch {epoch}, final loss = {losses[-1]:.4f}")


# @jax.jit
# def train_step(opt_state, params, X, Y):
#     loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
#     updates, opt_state = tx.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return loss, params, opt_state

# for i in tqdm(range(5000)):
#     key, train_key = jax.random.split(key)
#     xb, yb = get_batch(train_key, train_data)
#     loss, params, opt_state = train_step(opt_state, params, xb, yb)
#     if i % 1000 == 0:
#         key, est_key = jax.random.split(key)
#         losses = estimate_loss(est_key, params)
#         print(f"Training loss: {losses['train']}, Validation loss: {losses['val']}")

key, gen_key = jax.random.split(key)
print(decode(model.generate(gen_key, params, jnp.zeros((1,1), dtype=jnp.int32), 1000)[0].tolist()))

# TODO:
# - Validation losses
# - Jaxify the training loop with jax.lax for insane speeds
# - Add dropout?