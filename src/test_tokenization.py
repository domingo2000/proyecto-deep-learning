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

    def test_each_word_of_vocablary(self):
        input_tokenizer = InputTokenizer()

        vocabulary = [
            "after",  # 0
            "twice",  # 1
            "thrice",  # 2
            "opposite",  # 3
            "around",  # 4
            "left",  # 5
            "right",  # 6
            "turn",  # 7
            "walk",  # 8
            "look",  # 9
            "run",  # 10
            "jump",  # 11
            "<SOS>",  # 12
            "<EOS>",  # 13
            "<PAD>",  # 14
        ]

        for i, word in enumerate(vocabulary):
            input = word
            encoding = input_tokenizer.encode(input)
            assert encoding == [i]

    def test_some_input_cases(self):
        input_tokenizer = InputTokenizer()

        x_i = "<SOS> jump left <EOS>"

        tokenized_x_i = input_tokenizer.encode(x_i)

        assert tokenized_x_i == [12, 11, 5, 13]

        x_i = "<SOS> turn opposite right <EOS> <PAD> <PAD>"

        tokenized_x_i = input_tokenizer.encode(x_i)

        assert tokenized_x_i == [12, 7, 3, 6, 13, 14, 14]
