import torch
from model.performer import FastMultiheadAttention


def test_fast_attention():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4
    num_feature = 2

    query = torch.rand(batch_size, sequence_length, hidden_dim)
    key = torch.rand(batch_size, sequence_length, hidden_dim)
    value = torch.rand(batch_size, sequence_length, hidden_dim)

    fast_multihead_attention = FastMultiheadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_feature=num_feature,
    )

    outputs = fast_multihead_attention(
        query=query,
        key=key,
        value=value,
    )

    assert outputs.size() == (batch_size, sequence_length, hidden_dim)
