class Preprocessor:
    def __init__(self, context):
        if not context:
            raise NameError("Context was not given in initialization")

        self.context = context

    def transform(self, input):
        transform = ["<SOS>"]
        transform.extend(input.split(" "))

        transform.append("<EOS>")

        while len(transform) < (self.context):
            transform.append("<PAD>")

        transform = transform[0 : self.context]
        transform = " ".join(transform)
        return transform

    def input_transform(self, input):
        transform = []
        transform.extend(input.split(" "))

        while len(transform) < (self.context):
            transform.append("<PAD>")

        transform = transform[0 : self.context]
        transform = " ".join(transform)
        return transform

    def output_transform(self, input, sos=False, eos=True):
        transform = []

        if sos:
            transform.append("<SOS>")

        if input != "":
            transform.extend(input.split(" "))

        if eos:
            transform.append("<EOS>")

        while len(transform) < (self.context):
            transform.append("<PAD>")

        transform = transform[0 : self.context]
        transform = " ".join(transform)
        return transform


class NoPadPreprocessor:
    def __init__(self):
        pass

    def transform(self, input, sos=False, eos=False, shift=False):
        transform = []

        if sos:
            transform.append("<SOS>")

        if input != "":
            transform.extend(input.split(" "))

        if eos:
            transform.append("<EOS>")

        if shift:
            transform = transform[1:]
            transform.append("<PAD>")

        transform = " ".join(transform)
        return transform
