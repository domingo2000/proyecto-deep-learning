import os
import torch

"""
Simple Script to evaluate trained models with different datasets
"""

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

weights = torch.load("models/2023-07-03 17:26:30.826091/model-15.pt")

# Experiment 1
# train_path = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")
# test_path = os.path.join("SCAN", "simple_split", "tasks_test_simple.txt")

# Experiment 2

train_path = os.path.join("SCAN", "add_prim_split", "tasks_train_addprim_jump.txt")
test_path = os.path.join("SCAN", "add_prim_split", "tasks_test_addprim_jump.txt")

input_preprocessor = Preprocessor(eos=True)
output_preprocessor = Preprocessor(sos=True, eos=True)

input_tokenizer = Tokenizer(vocabulary=INPUT_VOCABULARY)
output_tokenizer = Tokenizer(vocabulary=OUTPUT_VOCABULARY)

train_dataset = SCANDataset(
    path=train_path,
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

test_dataset = SCANDataset(
    path=test_path,
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

model.load_state_dict(weights)

evaluator = Evaluator(output_tokenizer)

test_accuracy = evaluator.evaluate(model, dataset=test_dataset, show_progress=True)
print(f"Test Accuracy: {test_accuracy}")
train_accuracy = evaluator.evaluate(model, dataset=train_dataset, show_progress=True)
print(f"Train Accuracy: {train_accuracy}")


print(f"Accuracy | train: {train_accuracy}, test: {test_accuracy}")
