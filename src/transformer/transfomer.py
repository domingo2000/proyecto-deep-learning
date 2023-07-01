import torch
import torch.nn as nn
import parameters as P


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_size,
        tgt_size,
        dim_hidden,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feed_forward,
        dropout=0.0,
    ):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Embedding(src_size, dim_hidden)
        self.output_embedding = nn.Embedding(tgt_size, dim_hidden)

        self.input_position_embedding = nn.Embedding(src_size, dim_hidden)
        self.target_position_embedding = nn.Embedding(tgt_size, dim_hidden)

        self.transformer = nn.Transformer(
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=dim_hidden,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
            batch_first=True,
        )

        self.linear_feed_forward = nn.Linear(dim_hidden, tgt_size)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)
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

    def generate(self, x_i, sos_token, eos_token):
        device = x_i.device
        output = [sos_token]
        current_target_idx = 0
        while eos_token not in output:
            target_sequence = torch.tensor(
                output,
                device=device,
            )
            y_pred_logits = self.forward(x_i, target_sequence)
            y_pred = torch.multinomial(y_pred_logits, num_samples=1)

            next_token = y_pred[current_target_idx].item()
            current_target_idx += 1

            if current_target_idx >= P.MAX_LENGTH:
                break

            output.append(next_token)

        return output
