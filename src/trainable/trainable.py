from ray import air, tune
from transformer.transfomer import TransformerModel
from dataset.data import SCANDataset
from tokenization.tokenization import Tokenizer
from preprocessing.preprocessing import Preprocessor
from evaluation.evaluation import Evaluator

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import os

# import random

import parameters as P


class TrainTransformer(tune.Trainable):
    # Sets up the model as an independent, parallelizable process
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.gradient_clipping = config["gradient_clipping"]

        # # Set seeds for reproducibility
        # torch.manual_seed(0)
        # random.seed(0)

        # Setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Preprocessing
        input_preprocessor = Preprocessor(eos=True)
        output_preprocesor = Preprocessor(sos=True, eos=True)

        # Tokenization
        input_tokenizer = Tokenizer(P.INPUT_VOCABULARY)
        output_tokenizer = Tokenizer(P.OUTPUT_VOCABULARY)

        # Had to use absolute paths because of Ray (there must be a better way)
        prefix = "/home/domingo/deep-learning/proyecto-deep-learning/"
        if config["exp_type"] == "iid":
            train_path = os.path.join(prefix, "SCAN/simple_split/tasks_test_simple.txt")
            test_path = os.path.join(prefix, "SCAN/simple_split/tasks_test_simple.txt")
        elif config["exp_type"] == "ood":
            train_path = os.path.join(
                prefix, "SCAN/add_prim_split/tasks_test_addprim_jump.txt"
            )
            test_path = os.path.join(
                prefix, "SCAN/add_prim_split/tasks_train_addprim_jump.txt"
            )
        elif config["exp_type"] == "toy":
            train_path = os.path.join(prefix, "data/tasks_toy.txt")
            test_path = os.path.join(prefix, "data/tasks_toy.txt")

        self.train_dataset = SCANDataset(
            path=train_path,
            transform=lambda x: torch.tensor(
                input_tokenizer.encode(input_preprocessor.transform(x)), device=device
            ),
            target_transform=lambda x: torch.tensor(
                output_tokenizer.encode(output_preprocesor.transform(x)), device=device
            ),
            label_transform=lambda x: torch.tensor(
                output_tokenizer.encode(output_preprocesor.transform(x, shift=True)),
                device=device,
            ),
        )

        self.test_dataset = SCANDataset(
            path=test_path,
            transform=lambda x: torch.tensor(
                input_tokenizer.encode(input_preprocessor.transform(x)), device=device
            ),
            target_transform=lambda x: torch.tensor(
                output_tokenizer.encode(output_preprocesor.transform(x)), device=device
            ),
            label_transform=lambda x: torch.tensor(
                output_tokenizer.encode(output_preprocesor.transform(x, shift=True)),
                device=device,
            ),
        )

        # Model
        self.model = TransformerModel(
            src_size=P.INPUT_VOCAB_SIZE,
            tgt_size=P.OUTPUT_VOCAB_SIZE,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            num_encoder_layers=P.N_ENCODER_LAYERS,
            num_decoder_layers=P.N_DECODER_LAYERS,
            dim_feed_forward=P.D_FEED_FORWARD,
            dropout=config["dropout"],
            positional_encoding=config["positional_encoding"],
        )

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=config["learning_rate"])

        # Loss Criterion
        self.criterion = CrossEntropyLoss()

        # Evaluation
        self.evaluation = Evaluator(output_tokenizer, 32)
        self.final_evaluator = Evaluator(output_tokenizer, 1024)

        # Train
        self.model.train()
        self.model.to(device=device)

    # Defines an epoch. Returns a dictionary of metrics.
    def step(self):
        epoch_loss = 0
        for i, s_i in enumerate(self.train_dataset):
            x, y, y_label = s_i

            logits = self.model(x, y)

            self.optimizer.zero_grad()

            loss = self.criterion(logits, y_label)

            loss.backward()
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )  # Clipping to avoid exploding gradients
            self.optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(self.train_dataset)

        # Save both train and test accuracy and loss on the final epoch (P.EPOCHS)
        if self.iteration == P.EPOCHS:
            accuracy_train = self.final_evaluator.evaluate(
                self.model, self.train_dataset
            )
            accuracy_test = self.final_evaluator.evaluate(self.model, self.test_dataset)
            metrics = {
                "accuracy_train": accuracy_train,
                "accuracy_test": accuracy_test,
                "loss": epoch_loss,
            }
            return metrics

        # Save train accuracy and loss every P.EPOCH_N_METRICS epochs
        if self.iteration % P.EPOCH_N_METRICS == 0:
            accuracy = self.evaluation.evaluate(self.model, self.train_dataset)
            # Log accuracy and loss on the training set
            metrics = {"accuracy": accuracy, "loss": epoch_loss}
            return metrics

        # Save loss every epoch
        return {"loss": epoch_loss}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
