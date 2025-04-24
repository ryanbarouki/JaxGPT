import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

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

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
n_emb = 32
num_heads = 4
learning_rate = 1e-3

def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generate random starting positions for the examples
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

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
    
class FeedForward(nn.Module):
    n_emb: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(n_emb)(x)
        x = nn.relu(x)
        return x

class AttentionHead(nn.Module):
    head_size: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        k = nn.Dense(self.head_size, use_bias=False)(x) # (B, T, C) @ (C, hs) -> (B, T, hs)
        q = nn.Dense(self.head_size, use_bias=False)(x) # (B, T, C) @ (C, hs) -> (B, T, hs)
        v = nn.Dense(self.head_size, use_bias=False)(x) # (B, T, C) @ (C, hs) -> (B, T, hs)

        # Compute affinities
        wei = q @ k.swapaxes(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        tril = jnp.tril(jnp.ones((block_size, block_size))) # TODO: probably shouldn't be making this in every call
        mask = tril[:T, :T] == 0
        mask = mask[None, :, :]
        wei = jnp.where(mask, -jnp.inf, wei)
        wei = jax.nn.softmax(wei,axis=-1)

        return wei @ v # (B, T, T) @ (B, T, hs) ---> (B, T, hs)

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x):
        return jnp.concatenate([AttentionHead(self.head_size)(x) for _ in range(self.num_heads)], axis=-1) # (B, T, hs*num_heads)

class GPT(nn.Module):
    n_emb: int
    vocab_size: int
    block_size: int


    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        x = nn.Embed(self.vocab_size, self.n_emb)(idx) # (B, T, C)
        # positional embedding layer
        x += nn.Embed(self.block_size, self.n_emb)(jnp.arange(T)) # (B, T, C) + (T, C) (broadcast)
        x = MultiHeadAttention(num_heads, n_emb//num_heads)(x) # (B, T, C)
        x = FeedForward(n_emb)(x) # (B, T, C)
        logits = nn.Dense(self.vocab_size)(x) # (B, T, vocab_size)
        return logits

    def generate(self, key, params, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # crop idx so that it's never bigger than block_size (B, T)
            logits = self.apply(params, idx_cond)
            logits = logits[:, -1, :]
            key, subkey = jax.random.split(key)
            idx_next = jax.random.categorical(subkey, logits, shape=(logits.shape[0],1))
            idx = jnp.concatenate([idx, idx_next], axis=1).astype(int)
        return idx

model = GPT(n_emb=n_emb, vocab_size=vocab_size, block_size=block_size)

def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return optax.losses.softmax_cross_entropy_with_integer_labels(logits, y).mean()

key, init_key = jax.random.split(jax.random.PRNGKey(420696969))
xb, yb = get_batch(key, 'train')
params = model.init(init_key, jnp.ones_like(xb))

@jax.jit
def train_step(opt_state, params, X, Y):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, Y)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

# Training =======================================
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
key, gen_key = jax.random.split(key)
print(decode(model.generate(gen_key, params, jnp.zeros((1,1), dtype=jnp.int32), 500)[0].tolist())) # TODO SLOWWWWWWWWW
for i in range(10000):
    key, train_key = jax.random.split(key)
    xb, yb = get_batch(train_key, 'train')
    loss, params, opt_state = train_step(opt_state, params, xb, yb)
    if i % 1000 == 0:
        print(loss)
print(loss)

key, gen_key = jax.random.split(key)
print(decode(model.generate(gen_key, params, jnp.zeros((1,1), dtype=jnp.int32), 500)[0].tolist()))
# keys = jax.random.split(gen_key, num=100+1)
# gen_key, sample_keys = keys[0], keys[1:]
# samples = jax.vmap(lambda k: model.generate(k, params, jnp.zeros((1,1)), 100))(sample_keys)
