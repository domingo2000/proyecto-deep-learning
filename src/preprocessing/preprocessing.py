class Preprocessor:
    def __init__(self, context=None, pad=False, sos=False, eos=False):
        self.context = context
        self.pad = pad
        self.sos = sos
        self.eos = eos

    def transform(self, input, shift=False):
        transform = []

        if self.sos:
            transform.append("<SOS>")

        transform.extend(input.split())

        if self.eos:
            transform.append("<EOS>")

        if self.pad and self.context:
            while len(transform) < (self.context):
                transform.append("<PAD>")

        if shift:
            transform = transform[1:]
            transform.append("<PAD>")

        transform = " ".join(transform)
        return transform
