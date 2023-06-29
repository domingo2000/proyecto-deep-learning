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
