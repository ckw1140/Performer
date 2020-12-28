from torch import nn

from model.utils import (
    denominator,
    gaussian_orthogonal_random_matrix,
    gelu,
    linear_attention,
    nonnegative_softmax_kernel_feature_creator,
)


class FastMultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_features,
        scaling=0,
        qr_uniform_q=False,
    ):
        super(FastMultiheadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)
        self.WO = nn.Linear(hidden_dim, hidden_dim)
        self.projection_matrix = gaussian_orthogonal_random_matrix(
            num_rows=num_features,
            num_cols=self.head_dim,
            scaling=scaling,
            qr_uniform_q=qr_uniform_q,
        )

    def forward(self, query, key, value):
        batch_size = query.size()[0]

        # [batch_size, num_heads, sequence_length, head_dim]
        Q = self.split_heads(self.WQ(query), batch_size)
        K = self.split_heads(self.WK(key), batch_size)
        V = self.split_heads(self.WV(value), batch_size)

        # [batch_size, num_heads, sequence_length, num_features]
        Q_prime = nonnegative_softmax_kernel_feature_creator(
            data=Q,
            projection_matrix=self.projection_matrix,
        )
        K_prime = nonnegative_softmax_kernel_feature_creator(
            data=K,
            projection_matrix=self.projection_matrix,
        )
        denom = denominator(
            query=Q_prime,
            key=K_prime,
        )

        # [batch_size, num_heads, sequence_length, head_dim]
        attention_result = linear_attention(
            query=Q_prime,
            key=K_prime,
            value=V,
        )

        attention_result /= denom

        # [batch_size, sequence_length, hidden_dim]
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
        num_features,
        feed_forward_dropout_prob,
        layernorm_epsilon,
    ):
        super(Encoder, self).__init__()

        self.self_attention = FastMultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_features=num_features,
        )

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            dropout_prob=feed_forward_dropout_prob,
        )

        self.self_attention_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_epsilon)

    def forward(self, x):
        residual = x
        x = self.self_attention(x, x, x)
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
        self_attention_num_features,
        dec_enc_attention_num_features,
        feed_forward_dropout_prob,
        layernorm_epsilon,
    ):
        super(Decoder, self).__init__()

        self.self_attention = FastMultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_features=self_attention_num_features,
        )

        self.dec_enc_attention = FastMultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            featrue_dim=dec_enc_attention_num_features,
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

    def forward(self, x, enc_outputs):
        residual = x
        x = self.self_attention(x, x, x)
        x = self.self_attention_layernorm(x + residual)

        residual = x
        x = self.dec_enc_attention(x, enc_outputs, enc_outputs)
        x = self.dec_enc_attention_layernorm(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.feed_forward_layernorm(x + residual)

        return x
