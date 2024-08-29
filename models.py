import mlx.nn as nn
import mlx.core as mx

from mlx.utils import tree_flatten, tree_map


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.rope = nn.RoPE(dim_head, traditional=True, base=1e6)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def __call__(self, x, mask=None):
        b, n, d = x.shape
        x = self.norm(x)

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        reshaper = (lambda x: x.reshape(
            b, n, self.heads, -1).transpose(0, 2, 1, 3))
        queries, keys, values = map(reshaper, (queries, keys, values))

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        ).transpose(0, 2, 1, 3).reshape(b, n, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, mlp_dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.norm(x)
        return self.w2(self.dropout(nn.silu(self.w1(x))) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, seq_len, dropout):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self._mask = self._causal_mask(seq_len)

    def _causal_mask(self, n: int) -> mx.array:
        mask = mx.triu(mx.full(
            (n, n), -float('inf'), dtype=mx.float32
        ), k=1)
        return mask

    def __call__(self, x):
        x = x + self.attn(x, mask=self._mask)
        return x + self.ff(x)


class Transformer(nn.Module):
    def __init__(self, depth, dim, heads, n_tokens, seq_len, dropout=0., pool='cls'):
        super().__init__()
        assert pool in {'cls', 'mean'}, \
            'pool must be either cls (=) or mean (pooling)'
        self.pool = pool
        self.embedding = nn.Embedding(n_tokens, dim)
        self.layers = nn.Sequential(*[
            Block(dim, heads, dim//heads, dim*4, seq_len, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, n_tokens, bias=False)

    @property
    def num_params(self):
        return sum(x.size for k, x in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f'Number of parameters: {self.num_params}')

    def __call__(self, x):
        x = self.layers(self.embedding(x))
        x = mx.mean(x, axis=1) if self.pool == 'mean' else x[:, -1]
        return self.out(self.norm(x))


if __name__ == '__main__':
    n_tokens = 10
    seq_len = 5
    model = Transformer(depth=2,
                        dim=128,
                        heads=4,
                        n_tokens=n_tokens,
                        seq_len=seq_len,
                        dropout=0.1)
    model.summary()
    x = mx.random.randint(0, n_tokens, shape=(100, seq_len))
    y = model(x)
    print(y.shape)
