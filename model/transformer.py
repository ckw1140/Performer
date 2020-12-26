import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.utils import gelu


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_prob,
    ):
        super(MultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)
        self.WO = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
    ):
        batch_size = query.size()[0]

        # [batch_size, num_heads, sequence_length, head_dim]
        Q = self.split_heads(self.WQ(query), batch_size)
        K = self.split_heads(self.WK(key), batch_size)
        V = self.split_heads(self.WV(value), batch_size)

        # [batch_size, num_heads, sequence_length, sequence_length]
        scaled_dot_product = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(
            torch.tensor(self.head_dim).float()
        )
        scaled_dot_product += attention_mask.unsqueeze(1)
        attention_weight = F.softmax(scaled_dot_product, dim=-1)
        attention_weight = self.dropout(attention_weight)

        # [batch_size, num_heads, sequence_length, head_dim]
        attention_result = torch.matmul(attention_weight, V)

        # [batch_size, sequence_length, hidden_dim]
        attention_result = attention_result.transpose(1, 2).contiguous()
        attention_result = attention_result.view(batch_size, -1, self.hidden_dim)
        output = self.WO(attention_result)
        return output

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dropout_prob,
    ):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.dense1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        self_attention_dropout_prob,
        feed_forward_dropout_prob,
        layernorm_epsilon,
    ):
        super(Encoder, self).__init__()

        self.self_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_prob=self_attention_dropout_prob,
        )

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            dropout_prob=feed_forward_dropout_prob,
        )

        self.self_attention_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)

    def forward(self, x, attention_mask):
        residual = x
        x = self.self_attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
        )
        x = self.self_attention_layernorm(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.feed_forward_layernorm(x + residual)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        self_attention_dropout_prob,
        dec_enc_attention_dropout_prob,
        feed_forward_dropout_prob,
        layernorm_epsilon,
    ):
        super(Decoder, self).__init__()

        self.self_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_prob=self_attention_dropout_prob,
        )

        self.dec_enc_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_prob=dec_enc_attention_dropout_prob,
        )

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            dropout_prob=feed_forward_dropout_prob,
        )

        self.self_attention_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)
        self.dec_enc_attention_layernorm = nn.LayerNorm(
            hidden_dim, eps=layernorm_epsilon
        )
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)

    def forward(self, x, enc_outputs, self_attention_mask, dec_enc_attention_mask):
        residual = x
        x = self.self_attention(
            query=x,
            key=x,
            value=x,
            attention_mask=self_attention_mask,
        )
        x = self.self_attention_layernorm(x + residual)

        residual = x
        x = self.dec_enc_attention(
            query=x,
            key=enc_outputs,
            value=enc_outputs,
            attention_mask=dec_enc_attention_mask,
        )
        x = self.dec_enc_attention_layernorm(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.feed_forward_layernorm(x + residual)

        return x
