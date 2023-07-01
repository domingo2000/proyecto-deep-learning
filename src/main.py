from datetime import datetime

from transfomer import TransformerModel
from data import SCANDataset
from tokenization import InputTokenizer, OutputTokenizer
from preprocessing import Preprocessor

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score


import os
import random

import parameters as P

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
input_tokenizer = InputTokenizer()
output_tokenizer = OutputTokenizer()

#

# Dataset
train_path = os.path.join(*P.DATSET_PATH)
train_dataset = SCANDataset(
    path=train_path,
    transform=lambda x: torch.tensor(
        input_tokenizer.encode(input_preprocessor.transform(x)),
    ),
    target_transform=lambda x: torch.tensor(
        output_tokenizer.encode(output_preprocesor.transform(x))
    ),
    label_transform=lambda x: torch.tensor(
        output_tokenizer.encode(output_preprocesor.transform(x, shift=True))
    ),
)


# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

x, y, y_label = next(iter(train_dataloader))

# Model
model = TransformerModel()

# Optimizer
optimizer = Adam(model.parameters(), lr=P.LEARNING_RATE)

# Loss Criterion
criterion = CrossEntropyLoss()


# Train
model.train()
model.to(device=device)


def tokenization_sequence_to_string(tokenized_sequence, tokenizer=input_tokenizer):
    output = tokenizer.decode(tokenized_sequence)
    # output = output.replace(" <PAD>", "")
    # output = output.replace("<SOS> ", "")
    # output = output.replace("<SOS>", "")
    # output = output.replace(" <EOS>", "")
    # output = output.replace("<EOS>", "")
    return output


# Generator Function
def generate(x_i):
    eos_token = output_tokenizer.encode("<EOS>")[0]
    preprocessor = Preprocessor(sos=True)
    output = output_tokenizer.encode(preprocessor.transform(""))
    current_target_idx = 0
    while eos_token not in output:
        target_sequence = torch.tensor(
            output,
            device=device,
        )
        y_pred_logits = model.forward(x_i, target_sequence)
        y_pred = torch.multinomial(y_pred_logits, num_samples=1)

        next_token = y_pred[current_target_idx].item()
        current_target_idx += 1

        if current_target_idx >= P.MAX_LENGTH:
            break

        output.append(next_token)

    return output


def evaluate():
    model.eval()
    y_pred = []
    y_true = []
    sample = random.sample(list(iter(train_dataset)), k=32)
    for s_i in sample:
        x_i, y_i, y_label_i = s_i
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        y_label_i = y_label_i.to(device)
        output = generate(x_i)
        y_pred.append(output)
        y_true.append(y_i.tolist())

    y_pred = [output_tokenizer.decode(d) for d in y_pred]
    y_true = [output_tokenizer.decode(d) for d in y_true]
    acc = accuracy_score(y_true, y_pred)
    model.train()
    return acc


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


for epoch in range(P.EPOCHS):
    epoch_loss = 0
    for i, s_i in enumerate(train_dataset):
        x, y, y_label = s_i
        x = x.to(device)
        y = y.to(device)
        y_label = y_label.to(device)

        logits = model(x, y)

        # B, T, C = logits.shape
        # logits = logits.view(B * T, C)
        # y_label = y_label.view(B * T)

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
        x_test = train_dataset.transform("jump").to(device)
        x_target = train_dataset.target_transform("I_JUMP").to(device)
        logits = model(x_test, x_target)

        y_test = torch.multinomial(logits, num_samples=1)
        y_test = y_test.squeeze()
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # Jump Generated
        x_test = train_dataset.transform("jump").to(device)
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # run Generated
        x_test = train_dataset.transform("run").to(device)
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # walk and run Generated
        x_test = train_dataset.transform("walk and run").to(device)
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test.tolist(), tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        accuracy = evaluate()

        # Save model
        save_model(model, timestamp, epoch)

        print(f"| epoch: {epoch} | loss: {epoch_loss:.2f} | acc = {accuracy:.2f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
