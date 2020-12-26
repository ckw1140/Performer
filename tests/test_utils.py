import torch

from model.utils import (
    gaussian_orthogonal_random_matrix,
    linear_attention,
    nonnegative_softmax_kernel_feature_creator,
    orthogonal_matrix_chunk,
)


def test_nonnegative_softmax_kernel():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4
    projection_dim = 2

    head_dim = hidden_dim // num_heads
    data = torch.rand(batch_size, num_heads, sequence_length, head_dim)
    projection_matrix = torch.rand(projection_dim, head_dim)

    features = nonnegative_softmax_kernel_feature_creator(
        data=data,
        projection_matrix=projection_matrix,
    )

    assert features.size() == (batch_size, num_heads, sequence_length, projection_dim)


def test_orthogonal_matrix_chunk():
    num_cols = 5

    q = orthogonal_matrix_chunk(num_cols)
    assert q.size() == (num_cols, num_cols)


def test_gaussian_orthogonal_random_matrix():
    num_rows = 10
    num_cols = 5

    projection_matrix = gaussian_orthogonal_random_matrix(num_rows, num_cols)
    assert projection_matrix.size() == (num_rows, num_cols)


def test_linear_attention():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4

    head_dim = hidden_dim // num_heads
    query = torch.rand(batch_size, num_heads, sequence_length, head_dim)
    key = torch.rand(batch_size, num_heads, sequence_length, head_dim)
    value = torch.rand(batch_size, num_heads, sequence_length, head_dim)

    attention_result = linear_attention(query, key, value)
    assert attention_result.size() == (batch_size, num_heads, sequence_length, head_dim)
