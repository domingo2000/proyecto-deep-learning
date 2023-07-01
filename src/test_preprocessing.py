from preprocessing import Preprocessor, NoPadPreprocessor


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
        transformed_y_i = preprocessor.output_transform(y_i, sos=True, eos=True)
        assert transformed_y_i == "<SOS> I_TURN_RIGHT I_TURN_RIGHT <EOS>"


class TestNoPadPreprocessor:
    def test_no_pad_preprocessor(self):
        """
        Adds no padding when the context is longer
        """
        preprocessor = NoPadPreprocessor()

        x_i = "jump jump"
        y_i = "I_TURN_RIGHT I_TURN_RIGHT"
        transformed_y_i = preprocessor.transform(y_i, sos=True, eos=True)

        assert transformed_y_i == "<SOS> I_TURN_RIGHT I_TURN_RIGHT <EOS>"

        transformed_y_i = preprocessor.transform(y_i, sos=False, eos=True)
        transformed_x_i = preprocessor.transform(x_i, sos=False, eos=True)

        assert transformed_y_i == "I_TURN_RIGHT I_TURN_RIGHT <EOS>"
        assert transformed_x_i == "jump jump <EOS>"

        transformed_y_i = preprocessor.transform(y_i, sos=False, eos=False)

        assert transformed_y_i == "I_TURN_RIGHT I_TURN_RIGHT"

        transformed_y_i = preprocessor.transform(y_i, sos=True, eos=True, shift=True)

        assert transformed_y_i == "I_TURN_RIGHT I_TURN_RIGHT <EOS> <PAD>"
