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
    transform=lambda x: torch.tensor(input_tokenizer.encode(preprocessor.transform(x))),
    target_transform=lambda x: torch.tensor(
        output_tokenizer.encode(preprocessor.transform(x))
    ),
)


# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=P.BATCH_SIZE)

x, y = next(iter(train_dataloader))

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

    target_sequence = ["<PAD>" for i in range(P.CONTEXT_LENGTH - 1)]
    target_sequence.insert(0, "<SOS>")
    target_sequence = " ".join(target_sequence)
    target_sequence = output_tokenizer.encode(target_sequence)

    target_sequence = torch.tensor(target_sequence, device=device)

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
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x, y)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)

        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item()

        # if batch_n % 10 == 0:
        #     print(f"| batch: {batch_n} | loss: {loss:.2f} |")

    if epoch % 100 == 0:
        print(f"| epoch: {epoch} | loss: {epoch_loss:.2f} |")
        x_test = train_dataset.transform("jump").to(device)
        # x_test = x_test.reshape((1, *x_test.shape))
        x_target = train_dataset.target_transform("I_JUMP").to(device)
        # x_target = x_target.reshape((1, *x_target.shape))
        logits = model(x_test, x_target)  # generate(x_test)

        y_test = torch.multinomial(logits, num_samples=1)
        y_test = y_test.squeeze()
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )

        # Jump Twice
        x_test = train_dataset.transform("jump twice").to(device)
        # x_test = x_test.reshape((1, *x_test.shape))
        x_target = train_dataset.target_transform("I_JUMP I_JUMP").to(device)
        # x_target = x_target.reshape((1, *x_target.shape))
        logits = model(x_test, x_target)  # generate(x_test)

        y_test = torch.multinomial(logits, num_samples=1)
        y_test = y_test.squeeze()
        print(
            f"IN: {tokenization_sequence_to_string(x_test, tokenizer=input_tokenizer)}: OUT_PRED: {tokenization_sequence_to_string(y_test, tokenizer=output_tokenizer)}"
        )


# model.train()  # Turn on the train mode
# total_loss = 0.0
# start_time = time.time()
# src_mask = model.generate_square_subsequent_mask(bptt).to(device)
# for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
#     data, targets = get_batch(train_data, i)
#     optimizer.zero_grad()
#     if data.size(0) != bptt:
#         src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
#     output = model(data, src_mask)
#     loss = criterion(output.view(-1, ntokens), targets)
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#     optimizer.step()

#     total_loss += loss.item()
#     log_interval = 200
#     if batch % log_interval == 0 and batch > 0:
#         cur_loss = total_loss / log_interval
#         elapsed = time.time() - start_time
#         print(
#             "| epoch {:3d} | {:5d}/{:5d} batches | "
#             "lr {:02.2f} | ms/batch {:5.2f} | "
#             "loss {:5.2f} | ppl {:8.2f}".format(
#                 epoch,
#                 batch,
#                 len(train_data) // bptt,
#                 scheduler.get_last_lr()[0],
#                 elapsed * 1000 / log_interval,
#                 cur_loss,
#                 math.exp(cur_loss),
#             )
#         )
#         total_loss = 0
#         start_time = time.time()
