from preprocessing import input_transform, output_transform


class TestInputTransform:
    def test_input_transform(self):
        """
        Add <SOS> and <EOS>
        """
        x_i = "jump jump"
        transformed_x_i = input_transform(x_i, context=4)
        assert transformed_x_i == ["<SOS>", "jump", "jump", "<EOS>"]

    def test_input_transform_with_padding(self):
        """
        Applies padding: <PAD>  accordingly when the context is longer
        """
        x_i = "jump jump"
        transformed_x_i = input_transform(x_i, context=10)

        assert len(transformed_x_i) == 10
        assert transformed_x_i == [
            "<SOS>",
            "jump",
            "jump",
            "<PAD>",
            "<PAD>",
            "<PAD>",
            "<PAD>",
            "<PAD>",
            "<PAD>",
            "<EOS>",
        ]


class TestOutputTransform:
    def test_output_transform(self):
        """
        Add <SOS> and <EOS>
        """
        x_i = "I_TURN_RIGHT I_TURN_RIGHT"
        transformed_x_i = output_transform(x_i)
        assert transformed_x_i == ["<SOS>", "I_TURN_RIGHT", "I_TURN_RIGHT", "<EOS>"]

    def test_output_transform(self):
        """
        Adds no padding when the context is longer
        """
        y_i = "I_TURN_RIGHT I_TURN_RIGHT"
        transformed_y_i = output_transform(y_i)
        assert "<PAD>" not in transformed_y_i
