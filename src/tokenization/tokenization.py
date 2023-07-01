from parameters import INPUT_VOCABULARY, OUTPUT_VOCABULARY


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

        self.encoder = {v: i for i, v in enumerate(self.vocabulary)}
        self.decoder = {i: v for i, v in enumerate(self.vocabulary)}

    def encode(self, input):
        input = input.split(" ")
        tokenization = [self.encoder[d] for d in input]
        return tokenization

    def decode(self, encode):
        decoded = [self.decoder[d] for d in encode]
        return " ".join(decoded)
