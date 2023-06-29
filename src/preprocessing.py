def _base_transform(input: str, context=None, padding=False):
    transform = ["<SOS>"]
    transform.extend(input.split(" "))

    if padding:
        while len(transform) < (context - 1):
            transform.append("<PAD>")

    transform.append("<EOS>")
    return transform


def input_transform(input: str, context=10):
    return _base_transform(input, context=context, padding=True)


def output_transform(input: str):
    return _base_transform(input)
