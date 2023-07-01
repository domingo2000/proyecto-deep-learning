import pytest
from transfomer import TransformerModel
from tokenization import Tokenizer
from preprocessing import Preprocessor
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup
src_size = 10
tgt_size = 10
dim_hidden = 64
n_heads = 4
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.5
dim_feed_forward = 256

model = TransformerModel(
    src_size,
    tgt_size,
    dim_hidden,
    n_heads,
    num_encoder_layers,
    num_decoder_layers,
    dim_feed_forward,
    dropout,
)


class TestTransformer:
    def test_transformer_no_batch(self):
        src = torch.randint(0, src_size, (1, 10))[0]
        tgt = torch.randint(0, src_size, (1, 10))[0]

        logits = model(src, tgt)

        assert logits.shape == (10, tgt_size)

    def test_transformer_batch(self):
        batch_size = 32
        sequence_length = 5
        src = torch.randint(0, src_size, (batch_size, sequence_length))
        tgt = torch.randint(0, src_size, (batch_size, sequence_length))

        logits = model.forward(src, tgt)

        assert logits.shape == (batch_size, sequence_length, src_size)
