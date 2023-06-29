from tokenization import InputTokenizer
from data import SCANDataset
import os
import pytest


class TestInputTokenizer:
    """
    The tokenization should be

    and: 0
    after: 1
    twice: 2
    thrice: 3
    opposite: 4
    around: 5
    left: 6
    right: 7
    turn: 8
    walk: 9
    look: 10
    run: 11
    jump: 12
    <SOS>: 13
    <EOS>: 14
    <PAD>: 15
    """

    @pytest.mark.skip(reason="Not implemented")
    def test_each_word_of_vocablary(self):
        # Setup
        path = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")
        dataset = SCANDataset(path=path)
        input_tokenizer = InputTokenizer(data=dataset)

        vocabulary = [
            "after",
            "twice",
            "thrice",
            "opposite",
            "around",
            "left",
            "right",
            "turn",
            "walk",
            "look",
            "run",
            "jump",
            "<SOS>",
            "<EOS>",
            "<PAD>",
        ]

        for word, i in enumerate(vocabulary):
            input_tokenizer.encode()
            input = [word]
            assert input_tokenizer.encode(input) == i

    # def test_some_input_cases(self):
    #     x_i = ["<SOS>", "jump", "left", "<EOS>"]

    #     tokenized_x_i = tokenize_input(x_i)

    #     assert tokenized_x_i == (13, 12, 6, 14)
