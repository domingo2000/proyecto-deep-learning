import gradio as gr

import torch
from transformer.transfomer import TransformerModel
from preprocessing.preprocessing import Preprocessor
from tokenization.tokenization import Tokenizer
import parameters as P

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = torch.load(
    "runs/grid_search/TrainTransformer_d4eeb_00001_1_d_model=16,dropout=0.0000,exp_type=toy,gradient_clipping=0.5000,learning_rate=0.0001,n_heads=2,posi_2023-07-05_11-15-52/checkpoint_000010/model.pth"
)

model = TransformerModel(
    src_size=P.INPUT_VOCAB_SIZE,
    tgt_size=P.OUTPUT_VOCAB_SIZE,
    d_model=16,
    n_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feed_forward=256,
    dropout=0.0,
    positional_encoding="original",
)

model.load_state_dict(weights)
model.to(device=device)

input_preprocessor = Preprocessor(eos=True)
input_tokenizer = Tokenizer(P.INPUT_VOCABULARY)
output_tokenizer = Tokenizer(P.OUTPUT_VOCABULARY)
sos_token = output_tokenizer.encode("<SOS>")[0]
eos_token = output_tokenizer.encode("<EOS>")[0]


def inference(command):
    x_i = torch.tensor(
        input_tokenizer.encode(input_preprocessor.transform(command)),
        device=device,
    )
    output = model.generate(x_i, sos_token, eos_token)
    output = output_tokenizer.decode(output)
    return output


demo = gr.Interface(fn=inference, inputs="text", outputs="text")

demo.launch()
