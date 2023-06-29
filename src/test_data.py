from data import SCANDataset
from torch.utils.data import DataLoader
from preprocessing import Preprocessor
import pytest
import os


class TestDataset:
    def test_dataset_train(self):
        path = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")
        dataset = SCANDataset(path=path)

        s1 = (
            "jump opposite right twice and turn opposite right thrice",
            "I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT",
        )

        s_last = (
            "look opposite right twice after jump around right twice",
            "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_TURN_RIGHT I_LOOK",
        )

        assert dataset[0] == s1
        assert dataset[-1] == s_last
        assert len(dataset) == 16_728

    def test_dataset_test(self):
        path = os.path.join("SCAN", "simple_split", "tasks_test_simple.txt")
        dataset = SCANDataset(path=path)

        s1 = (
            "turn opposite right thrice and turn opposite left",
            "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_TURN_LEFT",
        )

        s_last = (
            "turn opposite right thrice and run opposite left",
            "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_TURN_LEFT I_RUN",
        )

        assert dataset[0] == s1
        assert dataset[-1] == s_last
        assert len(dataset) == 4_182

    def test_dataloader(self):
        path_train = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")
        train_dataset = SCANDataset(path=path_train)

        path_test = os.path.join("SCAN", "simple_split", "tasks_test_simple.txt")
        test_dataset = SCANDataset(path=path_test)

        s1_train = (
            "jump opposite right twice and turn opposite right thrice",
            "I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT",
        )

        s1_test = (
            "turn opposite right thrice and turn opposite left",
            "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_TURN_LEFT",
        )

        BATCH_SIZE = 32

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        train_x, train_y = next(iter(train_dataloader))

        test_x, test_y = next(iter(test_dataloader))

        assert len(train_x) == 32
        assert train_x[0] == s1_train[0]
        assert train_y[0] == s1_train[1]

        assert len(test_x) == 32
        assert test_x[0] == s1_test[0]
        assert test_y[0] == s1_test[1]

    def test_dataset_with_transform(self):
        """
        Context = 15
        """
        path = os.path.join("SCAN", "simple_split", "tasks_train_simple.txt")

        preprocessor = Preprocessor(context=15)
        dataset = SCANDataset(
            path=path,
            transform=lambda x: preprocessor.transform(x),
            target_transform=lambda y: preprocessor.transform(y),
        )

        s1 = (
            "<SOS> jump opposite right twice and turn opposite right thrice <EOS> <PAD> <PAD> <PAD> <PAD>",
            "<SOS> I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT <EOS> <PAD>",
        )

        assert dataset[0] == s1

        # Now test dataloader works correctly
        BATCH_SIZE = 32

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        x, y = next(iter(dataloader))

        assert len(x) == BATCH_SIZE
        assert x[0] == s1[0]
        assert y[0] == s1[1]
