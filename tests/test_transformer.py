import torch
from model.transformer import MultiheadAttention, FeedForward, Encoder, Decoder


def test_multihead_attention():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4
    dropout_prob = 0.3

    multihead_attention = MultiheadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    )

    query = torch.rand(batch_size, sequence_length, hidden_dim)
    key = torch.rand(batch_size, sequence_length, hidden_dim)
    value = torch.rand(batch_size, sequence_length, hidden_dim)
    attention_mask = (
        1 - torch.triu(torch.ones(sequence_length, sequence_length)).long().T
    )
    attention_mask = attention_mask.unsqueeze(0)

    outputs = multihead_attention(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
    )
    assert outputs.size() == (batch_size, sequence_length, hidden_dim)


def test_feed_forward():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    dropout_prob = 0.3

    feed_forward = FeedForward(
        hidden_dim=hidden_dim,
        dropout_prob=dropout_prob,
    )

    inputs = torch.rand(batch_size, sequence_length, hidden_dim)
    outputs = feed_forward(inputs)
    assert outputs.size() == (batch_size, sequence_length, hidden_dim)


def test_encoder():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4
    self_attention_dropout_prob = 0.3
    feed_forward_dropout_prob = 0.3
    layernorm_epsilon = 1e-6

    encoder = Encoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        self_attention_dropout_prob=self_attention_dropout_prob,
        feed_forward_dropout_prob=feed_forward_dropout_prob,
        layernorm_epsilon=layernorm_epsilon,
    )

    inputs = torch.rand(batch_size, sequence_length, hidden_dim)
    attention_mask = (
        1 - torch.triu(torch.ones(sequence_length, sequence_length)).long().T
    )
    attention_mask = attention_mask.unsqueeze(0)

    outputs = encoder(inputs, attention_mask)
    assert outputs.size() == (batch_size, sequence_length, hidden_dim)


def test_decoder():
    batch_size = 8
    sequence_length = 16
    hidden_dim = 8
    num_heads = 4
    self_attention_dropout_prob = 0.3
    dec_enc_attention_dropout_prob = 0.3
    feed_forward_dropout_prob = 0.3
    layernorm_epsilon = 1e-6

    decoder = Decoder(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        self_attention_dropout_prob=self_attention_dropout_prob,
        dec_enc_attention_dropout_prob=dec_enc_attention_dropout_prob,
        feed_forward_dropout_prob=feed_forward_dropout_prob,
        layernorm_epsilon=layernorm_epsilon,
    )

    inputs = torch.rand(batch_size, sequence_length, hidden_dim)
    enc_outputs = torch.rand(batch_size, sequence_length, hidden_dim)
    self_attention_mask = (
        1 - torch.triu(torch.ones(sequence_length, sequence_length)).long().T
    )
    self_attention_mask = self_attention_mask.unsqueeze(0)

    dec_enc_attention_mask = (
        1 - torch.triu(torch.ones(sequence_length, sequence_length)).long().T
    )
    dec_enc_attention_mask = dec_enc_attention_mask.unsqueeze(0)

    outputs = decoder(inputs, enc_outputs, self_attention_mask, dec_enc_attention_mask)
    assert outputs.size() == (batch_size, sequence_length, hidden_dim)
