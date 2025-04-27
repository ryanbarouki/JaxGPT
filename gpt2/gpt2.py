import flax.linen as nn
import jax
import jax.numpy as jnp
import math
from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MultiHeadAttention(nn.Module):
    config: Config

    def setup(self):
        self.causal_mask = jnp.tril(jnp.ones((self.config.block_size, self.config.block_size), dtype=bool))\
                                    .reshape(1,1,self.config.block_size, self.config.block_size)

        self.c_attn = nn.Dense(3 * self.config.n_embd, name='c_attn') # (C, 3*C) # q,k,v all in one matrix
        self.c_proj = nn.Dense(self.config.n_embd, name='c_proj') 
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        # NOTE head size = n_embd / n_head SHOULD BE INT

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2) # q, k, v for all attention heads
        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head)).swapaxes(1,2) # (B, nh, T, hs)
        k = jnp.reshape(k, (B, T, self.n_head, C // self.n_head)).swapaxes(1,2) # (B, nh, T, hs)
        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head)).swapaxes(1,2) # (B, nh, T, hs)

        att = (q @ k.swapaxes(-2, -1)) / math.sqrt(k.shape[-1]) # (B, nh, T, T)
        mask = self.causal_mask[:,:,:T,:T] 
        att = jnp.where(~mask, -jnp.inf, att)
        att = nn.softmax(att, axis=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) ---> (B, nh, T, hs)
        y = y.swapaxes(1,2).reshape(B, T, C) # (B, T, nh, hs) -> (B, T, C)
        y = self.c_proj(y)
        return y

class FeedForward(nn.Module):
    n_embd: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.n_embd, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.n_embd, name='c_proj')(x) # FOR RESIDUAL PATHWAY
        return x

class Block(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        # NOTE the x + (stuff) is a RESIDUAL CONNECTION
        x = x + MultiHeadAttention(self.config, name='attn')(nn.LayerNorm(name='ln_1')(x)) # (B, T, C) num_heads of (n_emb//num_heads)-dimensional self-attention
        x = x + FeedForward(self.config.n_embd, name='mlp')(nn.LayerNorm(name='ln_2')(x)) # (B, T, C)
        return x

class Blocks(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        blocks = nn.Sequential([Block(self.config, name=str(i)) for i in range(self.config.n_layer)])
        return blocks(x)

class GPT2(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        x = nn.Embed(self.config.vocab_size, self.config.n_embd, name='wte')(idx) # (B, T, C)
        # positional embedding layer
        x += nn.Embed(self.config.block_size, self.config.n_embd, name='wpe')(jnp.arange(T)) # (B, T, C) + (T, C) (broadcast)

        x = Blocks(self.config, name='h')(x)
        x = nn.LayerNorm(name='ln_f')(x)
        # NOTE huggingface and GPT-2 stops here and the last linear layer is the embedding layer transposed!
        logits = nn.Dense(self.config.vocab_size, name='lm_head', use_bias=False)(x) # (B, T, vocab_size)
        return logits

    def generate(self, key, params, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # crop idx so that it's never bigger than block_size (B, T)
            logits = self.apply(params, idx_cond)
            logits = logits[:, -1, :]
            key, subkey = jax.random.split(key)
            idx_next = jax.random.categorical(subkey, logits, shape=(logits.shape[0],1))
            idx = jnp.concatenate([idx, idx_next], axis=1).astype(int)
        return idx

    @classmethod
    def from_pretrained(cls, model_name):
        # TODO other gpt models
        # but for now just the 124M parameter model
        assert model_name == 'gpt2'
        from transformers import FlaxGPT2LMHeadModel

        hf_model = FlaxGPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        hf_params = hf_model.params
        hf_params['params'] = hf_params.pop('transformer', hf_params['transformer'])

        def transpose_params(params):
            to_transpose = ['attn.c_attn.kernel', 'attn.c_proj.kernel', 'mlp.c_fc.kernel', 'mlp.c_proj.kernel']
            def transpose(params, prefix=""):
                for p in params:
                    key = f"{prefix}.{p}" if prefix != "" else p
                    if isinstance(params[p], dict):
                        transpose(params[p], key)
                    else:
                        if any(key.endswith(w) for w in to_transpose):
                            params[p] = jnp.transpose(params[p])
            transpose(params)

        transpose_params(hf_params)
        hf_params['params']['lm_head'] = {'kernel': jnp.transpose(hf_params['params']['wte']['embedding'])}
        return hf_params


