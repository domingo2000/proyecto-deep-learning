from transfomer import TransformerModel
from data import SCANDataset
from tokenization import InputTokenizer, OutputTokenizer
from preprocessing import Preprocessor

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

import os

import parameters as P

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preprocessing
preprocessor = Preprocessor(context=P.CONTEXT_LENGTH)

# Tokenization
input_tokenizer = InputTokenizer()
output_tokenizer = OutputTokenizer()

# Dataset
# train_path = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")
train_path = os.path.join("src", "tasks_toy.txt")
train_dataset = SCANDataset(
    path=train_path,
    transform=lambda x: torch.tensor(
        input_tokenizer.encode(preprocessor.input_transform(x)),
    ),
    target_transform=lambda x: torch.tensor(
        output_tokenizer.encode(preprocessor.output_transform(x, sos=True))
    ),
    label_transform=lambda x: torch.tensor(
        output_tokenizer.encode(preprocessor.output_transform(x, eos=True))
    ),
)


# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=P.BATCH_SIZE)

x, y, y_label = next(iter(train_dataloader))

print("")
print(x)
print(y)

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
    output = tokenizer.decode(tokenized_sequence.tolist())
    output = output.replace(" <PAD>", "")
    # output = output.replace("<SOS> ", "")
    # output = output.replace("<SOS>", "")
    # output = output.replace(" <EOS>", "")
    # output = output.replace("<EOS>", "")
    return output


# Generator Function
def generate(x_i):
    eos_token = output_tokenizer.encode("<EOS>")[0]
    target_sequence = torch.tensor(
        output_tokenizer.encode(preprocessor.output_transform("", sos=True, eos=False)),
        device=device,
    )
    current_target_idx = 0
    while eos_token not in target_sequence:
        y_pred_logits = model.forward(x_i, target_sequence)
        y_pred = torch.multinomial(y_pred_logits, num_samples=1)

        next_token = y_pred[current_target_idx].item()
        current_target_idx += 1

        if current_target_idx == P.CONTEXT_LENGTH:
            break

        target_sequence[current_target_idx] = next_token

    return target_sequence


for epoch in range(P.EPOCHS):
    epoch_loss = 0
    for batch_n, batch in enumerate(train_dataloader):
        x, y, y_label = batch
        x = x.to(device)
        y = y.to(device)
        y_label = y_label.to(device)

        logits = model(x, y)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y_label = y_label.view(B * T)

        loss = criterion(logits, y_label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item()

        # if batch_n % 10 == 0:
        #     print(f"| batch: {batch_n} | loss: {loss:.2f} |")

    if epoch % 100 == 0:
        print(f"| epoch: {epoch} | loss: {epoch_loss:.2f} |")
        x_test = train_dataset.transform("jump").to(device)
        x_target = train_dataset.target_transform("I_JUMP").to(device)
        logits = model(x_test, x_target)

        y_test = torch.multinomial(logits, num_samples=1)
        y_test = y_test.squeeze()
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # Jump Generated
        x_test = train_dataset.transform("jump").to(device)
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # Jump twice Generated

        # Without generation, passing all the input as training set
        print(f"| epoch: {epoch} | loss: {epoch_loss:.2f} |")
        x_test = train_dataset.transform("jump twice").to(device)
        x_target = train_dataset.target_transform("I_JUMP I_JUMP").to(device)
        logits = model(x_test, x_target)

        y_test = torch.multinomial(logits, num_samples=1)
        y_test = y_test.squeeze()
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # With generation, passing only <SOS>
        x_test = train_dataset.transform("jump twice").to(device)
        y_test = generate(x_test)
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )
