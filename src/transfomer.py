import torch
import torch.nn as nn
import parameters as P
from tokenization import OutputTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            dim_feedforward=P.D_FEED_FORWARD,
            batch_first=True,  # (BATCH, SEQUENCE_LENGTH, FEATURE_DIM)
            dropout=P.DROPOUT,
        )

        # Boolean mask for masked self attention on decoder
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

    def get_attention_mask(self, tgt):
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=tgt.shape[0], device=device
        )
        return mask

    def forward(self, src, tgt):
        # Create mask for all the padding in the ouput
        output_pad_token = 8
        tgt_key_padding_mask = tgt == output_pad_token

        input_pad_token = 15
        src_key_padding_mask = src == input_pad_token

        src_embedding = self.input_embedding(src)
        src_pos_embedding = self.input_position_embedding(src)
        src = src_embedding + src_pos_embedding
        tgt_embedding = self.output_embedding(tgt)
        tgt_pos_embedding = self.target_position_embedding(tgt)
        tgt = tgt_embedding + tgt_pos_embedding

        output = self.transformer(
            src,
            tgt,
            tgt_mask=self.get_attention_mask(tgt),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        output = self.linear_feed_forward(output)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    from data import SCANDataset
    from tokenization import InputTokenizer, OutputTokenizer
    from preprocessing import Preprocessor

    model = TransformerModel()

    BATCH_SIZE = 32

    src = torch.randint(0, P.INPUT_VOCAB_SIZE, size=(BATCH_SIZE, P.CONTEXT_LENGTH))
    tgt = torch.randint(0, P.OUTPUT_VOCAB_SIZE, size=(BATCH_SIZE, P.CONTEXT_LENGTH))
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
