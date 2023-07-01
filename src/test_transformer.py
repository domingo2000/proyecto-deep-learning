import pytest
from transfomer import TransformerModel
from tokenization import InputTokenizer, OutputTokenizer
from preprocessing import NoPadPreprocessor
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestTransformer:
    def test_transformer_no_batch(self):
        preprocesor = NoPadPreprocessor()
        input_tokenizer = InputTokenizer()
        output_tokenizer = OutputTokenizer()

        model = TransformerModel()

        input = "jump twice"
        output = "I_JUMP I_JUMP"

        x_i = preprocesor.transform(input)
        y_i = preprocesor.transform(output, sos=True, eos=True)
        y_label = preprocesor.transform(output, sos=True, eos=True, shift=True)

        x_i = input_tokenizer.encode(x_i)
        y_i = output_tokenizer.encode(y_i)
        y_label = output_tokenizer.encode(y_label)

        x_i = torch.tensor(x_i)
        y_i = torch.tensor(y_i)
        y_label = torch.tensor(y_label)

        model.to(device=device)
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        y_label = y_label.to(device)

        logits = model.forward(x_i, y_i)

        assert logits.shape == (4, 9)  # 4 predictions, vocab size = 9

    @pytest.mark.skip(reason="batch inputs can't have different sequnce length")
    def test_transformer_batch(self):
        preprocesor = NoPadPreprocessor()
        input_tokenizer = InputTokenizer()
        output_tokenizer = OutputTokenizer()

        model = TransformerModel()

        input_batch = ["jump twice", "jump"]
        output_batch = ["I_JUMP I_JUMP", "I_JUMP"]

        x_1 = torch.tensor(
            [
                input_tokenizer.encode(preprocesor.transform(input))
                for input in input_batch
            ],
            device=device,
        )
        y_1 = torch.tensor(
            [
                output_tokenizer.encode(
                    preprocesor.transform(output, sos=True, eos=True)
                )
                for output in output_batch
            ],
            device=device,
        )
        y_label_1 = torch.tensor(
            [
                output_tokenizer.encode(
                    preprocesor.transform(output, sos=True, eos=True, shift=True)
                )
                for output in output_batch
            ],
            device=device,
        )

        model.to(device=device)

        logits = model.forward(x_1, y_1)

        print(logits)
