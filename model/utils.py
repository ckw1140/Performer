"""
https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py

위의 두 코드를 참고하여 작성한 코드입니다.
"""

import numpy as np
import torch


def gelu(x):
    cdf = 0.5 * (
        1.0 + torch.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )
    return x * cdf


def nonnegative_softmax_kernel_feature_creator(
    data,
    projection_matrix,
    is_query,
    normalize_data=True,
    eps=1e-4,
):
    """
    Softmax Kernel 의 Feature Map 입니다.
    반환하는 Kernel Feature 의 모든 원소가 nonnegative 입니다.

    :param data: [batch_size, num_heads, sequence_length, head_dim] Tensor 입니다. (Queries 또는 Keys)
    :param projection_matrix: [projection_dim, head_dim] Random Matrix 입니다.
    :normalize_data: normalization 이 필요한지 여부를 나타냅니다.
    """

    data_normalizer = data.size()[-1] ** -0.25 if normalize_data else 1
    ratio = projection_matrix.size()[0] ** -0.5
    
    data_mod_shape = data.size()[:2] + projection_matrix.size()

    # [batch_size, num_heads, projection_dim, head_dim]
    data_thick_random_matrix = torch.zeros(data_mod_shape) + projection_matrix

    # [batch_size, num_heads, sequence_length, projection_dim]
    data_dash = torch.matmul(
        data_normalizer * data,
        data_thick_random_matrix.transpose(2, 3),
    )

    # [batch_size, num_heads, sequence_length]
    diag_data = data.square()
    diag_data = diag_data.sum(dim=-1)

    # [batch_size, num_heads, sequence_length, 1]
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)

    if is_query:
        data_dash = ratio * torch.exp(data_dash - diag_data - data_dash.max(dim=-1, keepdim=True).values + eps)
    else:
        data_dash = ratio * torch.exp(data_dash - diag_data - data_dash.max(dim=-1).values + eps)

    return data_dash


def gaussian_orthogonal_random_matrix(
    num_rows,
    num_cols,
):
    pass