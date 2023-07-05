import pytest
import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

from parameters import (
    INPUT_VOCABULARY,
    OUTPUT_VOCABULARY,
    INPUT_VOCAB_SIZE,
    OUTPUT_VOCAB_SIZE,
)

from dataset.data import SCANDataset
from preprocessing.preprocessing import Preprocessor
from tokenization.tokenization import Tokenizer
from transformer.transfomer import TransformerModel
from evaluation.evaluation import Evaluator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# @pytest.mark.skip(reason="Integration test takes more time (20s aprox)")
def test_integration():
    # Setup

    input_preprocessor = Preprocessor(eos=True)
    output_preprocessor = Preprocessor(sos=True, eos=True)

    input_tokenizer = Tokenizer(vocabulary=INPUT_VOCABULARY)
    output_tokenizer = Tokenizer(vocabulary=OUTPUT_VOCABULARY)

    path = os.path.join("data", "tasks_toy.txt")
    dataset = SCANDataset(
        path=path,
        transform=lambda src: torch.tensor(
            input_tokenizer.encode(input_preprocessor.transform(src)),
            device=device,
        ),
        target_transform=lambda tgt: torch.tensor(
            output_tokenizer.encode(output_preprocessor.transform(tgt)),
            device=device,
        ),
        label_transform=lambda tgt: torch.tensor(
            output_tokenizer.encode(output_preprocessor.transform(tgt, shift=True)),
            device=device,
        ),
    )

    model = TransformerModel(
        src_size=INPUT_VOCAB_SIZE,
        tgt_size=OUTPUT_VOCAB_SIZE,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feed_forward=256,
        dropout=0.0,
    )

    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)

    criterion = CrossEntropyLoss()

    evaluator = Evaluator(output_tokenizer)

    epochs = 5
    for epoch in range(epochs):
        for i, s_i in enumerate(dataset):
            x, y, y_label = s_i
            x = x.to(device)
            y = y.to(device)
            y_label = y_label.to(device)

            logits = model(x, y)

            loss = criterion(logits, y_label)

            loss.backward()

            # Gradient Clipping to prevent divergence of exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

    # Evaluate
    accuracy = evaluator.evaluate(model, dataset)
    assert accuracy >= 0.95
