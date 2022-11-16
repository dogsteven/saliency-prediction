from torch.nn import Module, Linear, Dropout
from .scaled_dot_product_attention import ScaledDotProductAttention
from typing import Tuple

__all__ = ["MultiHeadAttention"]

class MultiHeadAttention(Module):
    def __init__(
        self,
        d_qkv: Tuple[int, int, int],
        n_heads: int,
        d_model: int,
        d_output: int,
        dropout_rate: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        assert (d_model % n_heads == 0)
        d_q, d_k, d_v = d_qkv
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_output = d_output

        self.attention = ScaledDotProductAttention(dropout_rate = dropout_rate)
        self.W_q = Linear(in_features = d_q, out_features = self.d_model, bias = bias)
        self.W_k = Linear(in_features = d_k, out_features = self.d_model, bias = bias)
        self.W_v = Linear(in_features = d_v, out_features = self.d_model, bias = bias)
        self.W = Linear(in_features = self.d_model, out_features = self.d_output, bias = bias)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries, keys, values):
        # shape of queries: (batch, n_q, d_q)
        # shape of keys: (batch, n_kv, d_k)
        # shape of values: (batch, n_kv, d_v)
        # --------------------------------------
        # shape of output: (batch, n_q, d_o)
        # --------------------------------------
        #
        # where
        #
        # batch: batch size
        # n_q: number of queries
        # n_kv: number of key-value pairs
        # d_q: query dimension
        # d_k: key dimension
        # d_v: value dimension
        # d_o: output dimension

        queries = self.pack_heads(self.W_q(queries))  # shape (batch * n_heads, n_q, d_proj)
        keys = self.pack_heads(self.W_k(keys))  # shape (batch * n_heads, n_kv, d_proj)
        values = self.pack_heads(self.W_v(values))  # shape (batch * n_heads, n_kv, d_proj)

        output = self.attention(queries, keys, values)  # shape (batch * n_heads, n_q, d_proj)
        output = self.unpack_heads(output)  # shape (batch, n_q, d_model)
        output = self.W(output)  # shape (batch, n_q, d_o)
        output = self.dropout(output)  # shape (batch, n_q, d_o)

        return output

    def pack_heads(self, x):
        # shape of x: (batch, n, d_model)
        # --------------------------------------
        # shape of output: (batch * n_heads, n, d_proj)
        # --------------------------------------
        b, n, _ = x.shape
        d_proj = self.d_model // self.n_heads

        x = x.reshape(b, n, self.n_heads, d_proj)  # shape (batch, n, n_heads, d_proj)
        x = x.transpose(1, 2)  # shape (batch, n_heads, n, d_proj)
        x = x.reshape(b * self.n_heads, n, d_proj)  # shape (batch * n_heads, n, d_proj)
        return x

    def unpack_heads(self, x):
        # shape of x: (batch * n_heads, n, d_proj)
        # --------------------------------------
        # shape of output: (batch, n, d_model)
        # --------------------------------------
        bh, n, d_proj = x.shape
        b = bh // self.n_heads

        x = x.reshape(b, self.n_heads, n, d_proj)  # shape (batch, n_heads, n, d_proj)
        x = x.transpose(1, 2)  # shape (batch, n, n_heads, d_proj)
        x = x.reshape(b, n, self.d_model)  # shape (batch, n, d_model)
        return x

    def forward_for_visualization(self, queries, keys, values):
        queries = self.pack_heads(self.W_q(queries))
        keys = self.pack_heads(self.W_k(keys))
        values = self.pack_heads(self.W_v(values))

        output = self.attention.forward_for_visualization(queries, keys, values)
        output = self.unpack_heads(output)
        output = self.W(output)
        output = self.dropout(output)

        return output