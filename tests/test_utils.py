import torch
from torch.nn import functional as F

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


def test_linear_attention_shape():
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


def test_linear_attention_error():
    batch_size = 1
    sequence_length = 10000
    num_heads = 1
    head_dim = 8
    num_feature = 1000

    query = torch.normal(0, 1, size=(batch_size, num_heads, sequence_length, head_dim))
    key = torch.normal(0, 1, size=(batch_size, num_heads, sequence_length, head_dim))
    value = torch.normal(0, 1, size=(batch_size, num_heads, sequence_length, head_dim))
    
    projection_matrix = gaussian_orthogonal_random_matrix(
        num_rows=num_feature,
        num_cols=head_dim,
    )

    query_prime = nonnegative_softmax_kernel_feature_creator(
        data=query.transpose(1, 2),
        projection_matrix=projection_matrix,
    )
    key_prime = nonnegative_softmax_kernel_feature_creator(
        data=key.transpose(1, 2),
        projection_matrix=projection_matrix,
    )

    fast_attention_result = linear_attention(
        query=query_prime,
        key=key_prime,
        value=value.transpose(1, 2),
    )

    scaled_dot_product = torch.matmul(query, key.transpose(2, 3)) / torch.sqrt(
        torch.tensor(head_dim).float()
    )
    attention_weight = F.softmax(scaled_dot_product, dim=-1)

    exact_attention_result = torch.matmul(attention_weight, value)

    max_error = 0.5
    error = torch.abs(exact_attention_result - fast_attention_result)
    assert torch.max(error) < max_error
