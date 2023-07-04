from preprocessing.preprocessing import Preprocessor


class TestPreprocessor:
    def test_no_transform(self):
        preprocessor = Preprocessor()
        x_i = ""
        y_i = ""
        assert preprocessor.transform(x_i) == y_i

    def test_sos(self):
        preprocessor = Preprocessor(sos=True)
        x_i = ""
        y_i = "<SOS>"
        assert preprocessor.transform(x_i) == y_i

    def test_eos(self):
        preprocessor = Preprocessor(eos=True)
        x_i = ""
        y_i = "<EOS>"
        assert preprocessor.transform(x_i) == y_i

    def test_padding(self):
        preprocessor = Preprocessor(pad=True, context=3)
        x_i = ""
        y_i = "<PAD> <PAD> <PAD>"
        assert preprocessor.transform(x_i) == y_i

    def test_combined(self):
        preprocessor = Preprocessor(sos=True, eos=True, pad=True, context=5)
        x_i = ""
        y_i = "<SOS> <EOS> <PAD> <PAD> <PAD>"
        assert preprocessor.transform(x_i) == y_i
