import torch
from model.utils import nonnegative_softmax_kernel_feature_creator

def test_nonnegative_softmax_kernel():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 16
    num_heads = 4
    projection_dim = 1

    head_dim = hidden_dim // num_heads
    data = torch.rand(batch_size, num_heads, sequence_length, head_dim)
    projection_matrix = torch.rand(projection_dim, head_dim)

    features = nonnegative_softmax_kernel_feature_creator(
        data=data,
        projection_matrix=projection_matrix,
        is_query=True,
    )

    assert features.size() == (batch_size, num_heads, sequence_length, projection_dim)