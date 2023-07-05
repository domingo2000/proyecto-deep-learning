from datetime import datetime

from transformer.transfomer import TransformerModel
from dataset.data import SCANDataset
from tokenization.tokenization import Tokenizer
from preprocessing.preprocessing import Preprocessor
from evaluation.evaluation import Evaluator

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter


import os
import random

import parameters as P

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)

# Tensorboard for visualization of training
writer = SummaryWriter()

# Create timestamp for run name
timestamp = str(datetime.now())

# Preprocessing
input_preprocessor = Preprocessor(eos=True)
output_preprocesor = Preprocessor(sos=True, eos=True)

# Tokenization
input_tokenizer = Tokenizer(P.INPUT_VOCABULARY)
output_tokenizer = Tokenizer(P.OUTPUT_VOCABULARY)
sos_token = output_tokenizer.encode("<SOS>")[0]
eos_token = output_tokenizer.encode("<EOS>")[0]

# Dataset
train_path = os.path.join(*P.DATSET_PATH)
train_dataset = SCANDataset(
    path=train_path,
    transform=lambda x: torch.tensor(
        input_tokenizer.encode(input_preprocessor.transform(x)), device=device
    ),
    target_transform=lambda x: torch.tensor(
        output_tokenizer.encode(output_preprocesor.transform(x)), device=device
    ),
    label_transform=lambda x: torch.tensor(
        output_tokenizer.encode(output_preprocesor.transform(x, shift=True)),
        device=device,
    ),
)


# Model
model = TransformerModel(
    src_size=P.INPUT_VOCAB_SIZE,
    tgt_size=P.OUTPUT_VOCAB_SIZE,
    d_model=P.D_MODEL,
    n_heads=P.N_HEADS,
    num_encoder_layers=P.N_ENCODER_LAYERS,
    num_decoder_layers=P.N_DECODER_LAYERS,
    dim_feed_forward=P.D_FEED_FORWARD,
    dropout=P.DROPOUT if P.DROPOUT else 0.0,
)

# Optimizer
optimizer = Adam(model.parameters(), lr=P.LEARNING_RATE)

# Loss Criterion
criterion = CrossEntropyLoss()

# Evaluation
evaluation = Evaluator(output_tokenizer, 32)

# Train
model.train()
model.to(device=device)


def tokenization_sequence_to_string(tokenized_sequence, tokenizer=input_tokenizer):
    output = tokenizer.decode(tokenized_sequence)
    return output


def save_model(model, timestamp, epoch):
    # Create directory for model
    dir_path = os.path.join("models", timestamp)
    metadata_path = os.path.join(dir_path, "metadata.py")
    model_path = os.path.join(dir_path, f"model-{epoch}.pt")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save metadata of model
    if not os.path.exists(metadata_path):
        with open(f"{dir_path}/metadata.py", "w") as metadata_file, open(
            "src/parameters.py", "r"
        ) as parameters_file:
            for line in parameters_file:
                metadata_file.write(line)

    # Save model parameters
    torch.save(model.state_dict(), model_path)


def run_examples():
    """
    Run some simple inputs to see overall convergence of model
    """

    # jump Generated
    x_test = train_dataset.transform("jump").to(device)
    y_test = model.generate(x_test, sos_token, eos_token)
    print(
        f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
    )

    # run Generated
    x_test = train_dataset.transform("run").to(device)
    y_test = model.generate(x_test, sos_token, eos_token)
    print(
        f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
    )

    # walk and run Generated
    x_test = train_dataset.transform("walk and run").to(device)
    y_test = model.generate(x_test, sos_token, eos_token)
    print(
        f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
    )


for epoch in range(P.EPOCHS):
    epoch_loss = 0
    for i, s_i in enumerate(train_dataset):
        x, y, y_label = s_i

        logits = model(x, y)

        loss = criterion(logits, y_label)

        loss.backward()
        if P.GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), P.GRADIENT_CLIPPING
            )  # Clipping to avoid exploding gradients
        optimizer.step()

        epoch_loss += loss.item()

        if i % P.BATCH_LOGGING_N == 0:
            print(f"| s_i: {i:3d} | loss: {loss:.2f} |")

    epoch_loss = epoch_loss / len(train_dataset)

    if epoch % P.EPOCH_N_METRICS == 0:
        run_examples()
        accuracy = evaluation.evaluate(model, train_dataset)

        # Save model
        save_model(model, timestamp, epoch)

        print(f"| epoch: {epoch} | loss: {epoch_loss:.2f} | acc = {accuracy:.2f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
