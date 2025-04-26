import regex as re

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

class BasicTokenizer:
    def train(self, text, vocab_size):
        ids = list(text.encode('utf-8'))
        num_merges = vocab_size - 256
        self.merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print(f"merging {vocab[idx]} into a new token {idx}")
            self.merges[pair] = idx

    def encode(self, text):
      tokens = list(text.encode("utf-8"))
      # go through merges in the order they were generated
      for pair in self.merges:
          new_tokens = []
          i = 0
          while i < len(tokens):
              if i < len(tokens) - 1 and (tokens[i],tokens[i+1]) == pair:
                  new_tokens.append(self.merges[pair])
                  i += 2
                  continue
              new_tokens.append(tokens[i])
              i += 1
          tokens = new_tokens
      return tokens

    def decode(self, ids):
        rev_merge = {v:k for k,v in self.merges.items()}
        def rec(subids, out=[]):
            for id in subids:
                if id < 256:
                    out.append(id)
                else:
                    if id in rev_merge:
                        rec(rev_merge[id], out)
            return out
        out = rec(ids)
        out = b"".join([bytes([x]) for x in out])
        return out.decode('utf-8', errors='replace')


class RegexTokenizer:
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    compiled_patten = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text, vocab_size):
        broken_text = re.findall(self.compiled_patten, text)
        ids = [list(ch.encode('utf-8')) for ch in broken_text]

        num_merges = vocab_size - 256
        self.merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for ch_ids in ids:
                get_stats(ch_ids, stats)
            if not stats:
                print(stats, ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print(f"merging {vocab[idx]} into a new token {idx}")
            self.merges[pair] = idx

    def encode(self, text):
      tokens = list(text.encode("utf-8"))
      # go through merges in the order they were generated
      for pair in self.merges:
          new_tokens = []
          i = 0
          while i < len(tokens):
              if i < len(tokens) - 1 and (tokens[i],tokens[i+1]) == pair:
                  new_tokens.append(self.merges[pair])
                  i += 2
                  continue
              new_tokens.append(tokens[i])
              i += 1
          tokens = new_tokens
      return tokens

    def decode(self, ids):
        rev_merge = {v:k for k,v in self.merges.items()}
        def rec(subids, out=[]):
            for id in subids:
                if id < 256:
                    out.append(id)
                else:
                    if id in rev_merge:
                        rec(rev_merge[id], out)
            return out
        out = rec(ids)
        out = b"".join([bytes([x]) for x in out])
        return out.decode('utf-8', errors='replace')
