from tokenization.tokenization import Tokenizer
from parameters import INPUT_VOCABULARY, OUTPUT_VOCABULARY


class TestTokenizer:
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

    def test_each_word_of_input_vocablary(self):
        input_tokenizer = Tokenizer(vocabulary=INPUT_VOCABULARY)

        vocabulary = INPUT_VOCABULARY

        for i, word in enumerate(vocabulary):
            input = word
            encoding = input_tokenizer.encode(input)
            assert encoding == [i]

    def test_some_input_cases(self):
        input_tokenizer = Tokenizer(vocabulary=INPUT_VOCABULARY)

        x_i = "<SOS> jump left <EOS>"

        tokenized_x_i = input_tokenizer.encode(x_i)

        assert tokenized_x_i == [13, 12, 6, 14]

        x_i = "<SOS> turn opposite right <EOS> <PAD> <PAD>"

        tokenized_x_i = input_tokenizer.encode(x_i)

        assert tokenized_x_i == [13, 8, 4, 7, 14, 15, 15]

    def test_each_word_of_output_vocablary(self):
        output_tokenizer = Tokenizer(vocabulary=OUTPUT_VOCABULARY)

        vocabulary = OUTPUT_VOCABULARY

        for i, word in enumerate(vocabulary):
            input = word
            encoding = output_tokenizer.encode(input)
            assert encoding == [i]

    def test_some_output_cases(self):
        output_tokenizer = Tokenizer(vocabulary=OUTPUT_VOCABULARY)

        x_i = "<SOS> I_WALK I_RUN <EOS>"
        tokenized_x_i = output_tokenizer.encode(x_i)

        assert tokenized_x_i == [6, 0, 2, 7]

        x_i = "<SOS> I_TURN_RIGHT I_WALK I_JUMP <EOS> <PAD> <PAD>"

        tokenized_x_i = output_tokenizer.encode(x_i)

        assert tokenized_x_i == [6, 5, 0, 3, 7, 8, 8]
