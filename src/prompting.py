import gradio as gr

import torch
from transformer.transfomer import TransformerModel
from preprocessing.preprocessing import Preprocessor
from tokenization.tokenization import InputTokenizer, OutputTokenizer
import parameters as P

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = torch.load("models/2023-06-30 17:41:14.024773/model-47.pt")
model = TransformerModel()
model.load_state_dict(weights)
model.to(device=device)

preprocessor = Preprocessor()
input_tokenizer = InputTokenizer()
output_tokenizer = OutputTokenizer()


def generate(x_i):
    eos_token = output_tokenizer.encode("<EOS>")[0]

    output = output_tokenizer.encode(preprocessor.transform("", sos=True, eos=False))
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


def inference(command):
    x_i = torch.tensor(
        input_tokenizer.encode(preprocessor.transform(command, eos=True)), device=device
    )
    output = generate(x_i)
    output = output_tokenizer.decode(output)
    return output


demo = gr.Interface(fn=inference, inputs="text", outputs="text")

demo.launch()
