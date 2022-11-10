from torch import bmm, softmax
from torch.nn import Module, Dropout
from math import sqrt

__all__ = ["ScaledDotProductAttention"]

class ScaledDotProductAttention(Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries, keys, values):
        # shape of queries: (batch, n_q, d)
        # shape of keys: (batch, n_kv, d)
        # shape of values: (batch, n_kv, d_v)
        # --------------------------------------
        # shape of output: (batch, n_q, d_v)
        # --------------------------------------
        #
        # where
        #
        # batch: batch size
        # n_q: number of queries
        # n_kv: number of key-value pairs
        # d: input dimension
        # d_v: output dimension

        b, n_q, d = queries.shape

        scores = bmm(queries, keys.transpose(1, 2)) / sqrt(d)  # shape (batch, n_q, n_kv)
        scores = softmax(scores, 2)  # shape (batch, n_q, n_kv)

        output = bmm(scores, values)  # shape (batch, n_q, d_v)
        output = self.dropout(output)  # shape (batch, n_q, d_v)

        return output