from preprocessing import Preprocessor


class TestPreprocessor:
    def test_input_transform_no_padding(self):
        """
        Add <SOS> and <EOS>
        """
        preprocessor = Preprocessor(context=2)

        x_i = "jump jump"
        transformed_x_i = preprocessor.input_transform(x_i)

        assert transformed_x_i == "jump jump"

    def test_exceeded_context_crops_input(self):
        preprocessor = Preprocessor(context=4)

        x_i = "jump jump jump jump jump jump"
        transformed_x_i = preprocessor.input_transform(x_i)

        assert transformed_x_i == "jump jump jump jump"

    def test_transform_with_padding(self):
        """
        Applies padding: <PAD>  accordingly when the context is longer
        """
        preprocessor = Preprocessor(context=10)
        x_i = "jump jump"
        transformed_x_i = preprocessor.input_transform(x_i)

        assert len(transformed_x_i.split(" ")) == 10
        assert (
            transformed_x_i
            == "jump jump <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>"
        )

    def test_output_transform(self):
        """
        Add <SOS> and <EOS>
        """
        preprocessor = Preprocessor(context=4)

        y_i = "I_TURN_RIGHT I_TURN_RIGHT"
        transformed_y_i = preprocessor.output_transform(y_i)
        assert transformed_y_i == "<SOS> I_TURN_RIGHT I_TURN_RIGHT <EOS>"

    def test_output_transform_padding(self):
        """
        Adds no padding when the context is longer
        """
        preprocessor = Preprocessor(context=10)

        y_i = "I_TURN_RIGHT I_TURN_RIGHT"
        transformed_y_i = preprocessor.output_transform(y_i)

        assert len(transformed_y_i.split(" ")) == 10
        assert (
            transformed_y_i
            == "<SOS> I_TURN_RIGHT I_TURN_RIGHT <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>"
        )

    def test_output_transform_empty_sequence(self):
        preprocessor = Preprocessor(context=10)

        y_i = ""
        transformed_y_i = preprocessor.output_transform(y_i, eos=False)

        assert len(transformed_y_i.split(" ")) == 10
        assert (
            transformed_y_i
            == "<SOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>"
        )
