import mindspore
import mindspore.numpy as np
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from typing import Optional, Dict

# Self Attention
# Mindspore
class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads  # 向下取整
        self.scale = Tensor(head_dim ** -0.5)

        self.q = nn.Dense(dim, dim)
        self.k = nn.Dense(dim, dim)
        self.v = nn.Dense(dim, dim)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x_to_query, x_to_key, x_to_value, mask=None):
        """Attention construct."""
        bq, nq, cq = x_to_query.shape
        bk, nk, ck = x_to_key.shape
        bv, nv, cv = x_to_value.shape
        q = self.q(x_to_query)
        k = self.k(x_to_key)
        v = self.v(x_to_value)
        q = ops.reshape(q, (bq, nq, self.num_heads, cq // self.num_heads))  # (b, n, h, c/h)
        k = ops.reshape(k, (bk, nk, self.num_heads, ck // self.num_heads))
        v = ops.reshape(v, (bv, nv, self.num_heads, cv // self.num_heads))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        
        if mask is not None:
            attn = ops.masked_fill(attn, mask, -1e9)  # pad mask
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (bq, nq, cq))
        out = self.out(out)
        out = self.out_drop(out)

        return out
      
# 前馈层和残差连接模块
class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x

# Mindspore Transformer Encoder
class TransformerEncoderLayer(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoderLayer, self).__init__()
        self.normalization1 = norm((dim,))
        self.normalization2 = norm((dim,))
        self.attention = Attention(dim=dim,
                                   num_heads=num_heads,
                                   keep_prob=keep_prob,
                                   attention_keep_prob=attention_keep_prob)

        self.feedforward = FeedForward(in_features=dim,
                                       hidden_features=mlp_dim,
                                       activation=activation,
                                       keep_prob=keep_prob)
    def construct(self, x, src_key_padding_mask=None):
        out = x + self.attention(x, x, x, src_key_padding_mask)
        out = self.normalization1(out)
        out = out + self.feedforward(out)
        out = self.normalization2(out)
        return out
    
class TransformerEncoder(nn.Cell):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
    def construct(self, x, src_key_padding_mask=None):
        for _ in range(self.num_layers):
            x = self.layer(x, src_key_padding_mask)
        return x
      
# Mindspore Transformer Decoder
class TransformerDecoderLayer(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerDecoderLayer, self).__init__()
        normalization1 = norm((dim,))
        normalization2 = norm((dim,))
        normalization3 = norm((dim,))
        
        self.attention1 = Attention(dim=dim,
                                   num_heads=num_heads,
                                   keep_prob=keep_prob,
                                   attention_keep_prob=attention_keep_prob)
        self.attention2 = Attention(dim=dim,
                                   num_heads=num_heads,
                                   keep_prob=keep_prob,
                                   attention_keep_prob=attention_keep_prob)
        
        self.feedforward = FeedForward(in_features=dim,
                                       hidden_features=mlp_dim,
                                       activation=activation,
                                       keep_prob=keep_prob)
    def construct(self, tgt, memory, tgt_key_padding_mask=None):
        out = tgt + self.attention1(tgt, tgt, tgt, tgt_key_padding_mask)
        out = self.normalization1(out)
        out = out + self.attention2(out, memory, memory)
        out = self.normalization2(out)
        out = out + self.feedforward(out)
        out = self.normalization3(out)
        return out

class TransformerDecoder(nn.Cell):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
    def construct(self, tgt, memory, tgt_key_padding_mask=None):
        for _ in range(self.num_layers):
            tgt = self.layer(tgt, memory, tgt_key_padding_mask)
        return tgt
