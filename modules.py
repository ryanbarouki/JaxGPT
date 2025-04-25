import jax
import jax.numpy as jnp
import flax.linen as nn

block_size = 64

class FeedForward(nn.Module):
    n_emb: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.n_emb)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_emb)(x) # FOR RESIDUAL PATHWAY
        return x

class AttentionHead(nn.Module):
    head_size: int

    def setup(self):
        self.causal_mask = jnp.tril(jnp.ones((block_size, block_size), dtype=bool))

        self.to_q = nn.Dense(self.head_size, use_bias=False) # (C, hs)
        self.to_k = nn.Dense(self.head_size, use_bias=False) # (C, hs)
        self.to_v = nn.Dense(self.head_size, use_bias=False) # (C, hs)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.to_q(x) # (B, T, C) @ (C, hs) -> (B, T, hs)
        k = self.to_k(x) # (B, T, C) @ (C, hs) -> (B, T, hs)
        v = self.to_v(x) # (B, T, C) @ (C, hs) -> (B, T, hs)

        attn_scores = (q @ k.swapaxes(-2, -1)) * (self.head_size ** -0.5) # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        mask = self.causal_mask[:T, :T] 
        attn_scores = jnp.where(~mask[None, :, :], -jnp.inf, attn_scores)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        return attn_weights @ v # (B, T, T) @ (B, T, hs) ---> (B, T, hs)

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x):
        out = jnp.concatenate([AttentionHead(self.head_size)(x) for _ in range(self.num_heads)], axis=-1) # (B, T, hs*num_heads)
        out = nn.Dense(self.num_heads*self.head_size)(out) # FOR RESIDUAL PATHWAY
        return out

class Block(nn.Module):
    n_emb: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # NOTE the x + (stuff) is a RESIDUAL CONNECTION
        x = x + MultiHeadAttention(self.num_heads, self.n_emb//self.num_heads)(nn.LayerNorm()(x)) # (B, T, C) num_heads of (n_emb//num_heads)-dimensional self-attention
        x = x + FeedForward(self.n_emb)(nn.LayerNorm()(x)) # (B, T, C)
        return x

class GPT(nn.Module):
    n_emb: int
    vocab_size: int
    block_size: int
    num_heads: int
    num_blocks: int


    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        x = nn.Embed(self.vocab_size, self.n_emb)(idx) # (B, T, C)
        # positional embedding layer
        x += nn.Embed(self.block_size, self.n_emb)(jnp.arange(T)) # (B, T, C) + (T, C) (broadcast)
        blocks = nn.Sequential([Block(self.n_emb, self.num_heads) for _ in range(self.num_blocks)])
        x = blocks(x)
        logits = nn.Dense(self.vocab_size)(nn.LayerNorm()(x)) # (B, T, vocab_size)
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
