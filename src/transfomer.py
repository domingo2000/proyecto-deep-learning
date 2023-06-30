import torch
import torch.nn as nn
import parameters as P
from tokenization import OutputTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# based on https://torchtutorialstaging.z5.web.core.windows.net/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Embedding(P.INPUT_VOCAB_SIZE, P.D_MODEL)
        self.output_embedding = nn.Embedding(P.OUTPUT_VOCAB_SIZE, P.D_MODEL)

        self.input_position_embedding = nn.Embedding(P.INPUT_VOCAB_SIZE, P.D_MODEL)
        self.target_position_embedding = nn.Embedding(P.OUTPUT_VOCAB_SIZE, P.D_MODEL)

        self.transformer = nn.Transformer(
            nhead=P.N_HEADS,
            num_encoder_layers=P.N_ENCODER_LAYERS,
            num_decoder_layers=P.N_DECODER_LAYERS,
            d_model=P.D_MODEL,
            batch_first=True,  # (BATCH, SEQUENCE_LENGTH, FEATURE_DIM)
        )

        self.mask = nn.Transformer.generate_square_subsequent_mask(
            sz=P.CONTEXT_LENGTH, device=device
        )

        self.linear_feed_forward = nn.Linear(P.D_MODEL, P.OUTPUT_VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_feed_forward.bias.data.zero_()
        self.linear_feed_forward.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src_embedding = self.input_embedding(src)
        src_pos_embedding = self.input_position_embedding(src)
        src = src_embedding + src_pos_embedding
        tgt_embedding = self.output_embedding(tgt)
        tgt_pos_embedding = self.target_position_embedding(tgt)
        tgt = tgt_embedding + tgt_pos_embedding

        output = self.transformer(src, tgt, tgt_mask=self.mask)
        output = self.linear_feed_forward(output)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    model = TransformerModel()

    src = torch.randint(0, P.INPUT_VOCAB_SIZE, size=(P.BATCH_SIZE, P.CONTEXT_LENGTH))
    tgt = torch.randint(0, P.OUTPUT_VOCAB_SIZE, size=(P.BATCH_SIZE, P.CONTEXT_LENGTH))
    out = model(src, tgt)
    print(out)

    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    # B, T, C = batch size, sequence length, output_vocabulary_size
    B, T, C = out.shape
    out = out.view(B * T, C)
    tgt = tgt.view(B * T)
    loss = criterion(out, tgt)

    print(loss)
