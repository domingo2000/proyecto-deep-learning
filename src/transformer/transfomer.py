import torch
import torch.nn as nn
import parameters as P
import math


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.d_model = d_model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # Adjusted code for no batch
        seq_len = x.size(0)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Model

    Arguments:
        src_size: dimension of the input
        tgt-size: dimension of the output
        d_model: dimmension of hidden state of transformer
        n_heads: number of heads in multi head attention layers
        num_encoder_layers: number of layers in encoder
        num_decoder_layers: number of layers in decoder
        dim_feed_forward: dimension of the feed_forward out layer
        droput: dropout regularizaion [0, 1]
        positional_encodding: "original" || "embedding", switch to use sin/cos vs learned positional encoding in model
    """

    def __init__(
        self,
        src_size,
        tgt_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feed_forward,
        dropout=0.0,
        positional_encoding="original",  # or "embedding"
    ):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_size, d_model)

        if positional_encoding == "embedding":
            self.src_positional_encodding = nn.Embedding(src_size, d_model)
            self.tgt_positional_encoding = nn.Embedding(tgt_size, d_model)
        elif "original":
            self.src_positional_encodding = PositionalEncoding(d_model, dropout)
            self.tgt_positional_encoding = PositionalEncoding(d_model, dropout)
        else:
            raise NameError(
                "Positional Encoding not implemented, choose 'original' || 'embedding'"
            )

        self.transformer = nn.Transformer(
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
            batch_first=True,
        )

        self.linear_feed_forward = nn.Linear(d_model, tgt_size)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_feed_forward.bias.data.zero_()
        self.linear_feed_forward.weight.data.uniform_(-initrange, initrange)

    def get_attention_mask(self, tgt):
        if len(tgt.shape) == 3:
            tgt = tgt[0]
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=tgt.shape[0], device=tgt.device
        )
        return mask

    def forward(self, src, tgt):
        # Create mask for all the padding in the ouput
        output_pad_token = 8
        tgt_key_padding_mask = tgt == output_pad_token

        input_pad_token = 15
        src_key_padding_mask = src == input_pad_token

        src_embedding = self.src_embedding(src)
        src_pos_embedding = self.src_positional_encodding(src_embedding)
        src = src_embedding + src_pos_embedding
        tgt_embedding = self.tgt_embedding(tgt)
        tgt_pos_embedding = self.tgt_positional_encoding(tgt_embedding)
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

    def generate(self, src, sos_token, eos_token):
        device = src.device
        tgt = [sos_token]
        current_target_idx = 0
        while eos_token not in tgt:
            target_sequence = torch.tensor(
                tgt,
                device=device,
            )
            y_pred_logits = self.forward(src, target_sequence)
            y_pred = torch.multinomial(y_pred_logits, num_samples=1)

            next_token = y_pred[current_target_idx].item()
            current_target_idx += 1

            if current_target_idx >= P.MAX_LENGTH:
                break

            tgt.append(next_token)

        return tgt
