import random
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Evaluator:
    def __init__(self, output_tokenizer, sample_k: int = None):
        self.sample_k = sample_k
        self._output_tokenizer = output_tokenizer
        self._sos_token = output_tokenizer.encode("<SOS>")[0]
        self._eos_token = output_tokenizer.encode("<EOS>")[0]
        self._is_sampled = True if sample_k else False

    def evaluate(self, model, dataset, show_progress=False):
        """
        Evaluate the models and return the accuracy obtained
        """
        model.eval()

        y_pred = []
        y_true = []

        if self._is_sampled:
            k = len(dataset) if self.sample_k > len(dataset) else self.sample_k
            sample = random.sample(list(iter(dataset)), k=k)
        else:
            sample = dataset

        with torch.no_grad():
            for s_i in tqdm(sample, disable=not show_progress, mininterval=1):
                x_i, y_i, _ = s_i
                output = model.generate(x_i, self._sos_token, self._eos_token)
                y_pred.append(output)
                y_true.append(y_i.tolist())

        y_pred = [self._output_tokenizer.decode(d) for d in y_pred]
        y_true = [self._output_tokenizer.decode(d) for d in y_true]
        acc = accuracy_score(y_true, y_pred)
        model.train()

        return acc
