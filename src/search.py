import os
from parameters import EPOCHS, EPOCH_N_METRICS, CHECKPOINT_FREQUENCY

from trainable.trainable import TrainTransformer
from ray import air, tune
import ray

# Installation with !pip install -q -U ray[tune]


#### Hyperparameter Search with fault tolerance ####

# Set serialisation path
# storage_path = "/content/drive/MyDrive/ray_results" # Using Google Drive/Colab
storage_path = "./runs"
stop = {  # number of epochs (it should include both ends)
    "training_iteration": EPOCHS + 1
}

config_control = {
    "learning_rate": tune.grid_search([1e-5]),  # [1e-5, 1e-4, 1e-3]
    "d_model": tune.grid_search([64]),  # [16, 32, 64] ; Add 128?
    "n_heads": tune.grid_search([2]),  # [1, 2, 4]
    "dropout": tune.grid_search([0.0]),  # [0.0, 0.3] ; Add 0.5?
    "gradient_clipping": tune.grid_search([0.5]),  # [0, 0.5]
    "exp_type": tune.grid_search(  # ["iid", "ood"] or ["toy"] for testing the search
        ["toy"]
    ),
    "positional_encoding": tune.grid_search(["original"]),  # ["original", "embedding"]
}

sensibility_dict = {
    "learning_rate": tune.grid_search([1e-5, 1e-3]),  # [1e-5, 1e-4, 1e-3]
    "d_model": tune.grid_search([16, 32, 64]),  # [16, 32, 64]
    "n_heads": tune.grid_search([1, 2, 4]),  # [1, 2, 4]
    "dropout": tune.grid_search([0.0, 0.3]),  # [0.0, 0.3]
    # "gradient_clipping": tune.grid_search([0, 0.5]),  # [0, 0.5]
}

experiments = []

ray.init(num_gpus=1)


def build_experiments():
    experiments = ["iid", "ood"]
    models = ["original", "embedding"]

    ray_experiments = {}
    for experiment in experiments:
        for model in models:
            experiment_dir = f"{experiment}/{model}"

            # First we add thec control variables
            control_key = f"{experiment_dir}/control"
            ray_experiments[control_key] = {
                "run": TrainTransformer,
                "config": {
                    **config_control,
                    "exp_type": tune.grid_search([experiment]),
                    "positional_encoding": tune.grid_search([model]),
                },
                "local_dir": storage_path,
                "stop": stop,
                "checkpoint_config": air.CheckpointConfig(
                    checkpoint_frequency=CHECKPOINT_FREQUENCY, checkpoint_at_end=True
                ),  # set num_to_keep=3 to save disk space
                "num_samples": 1,  # Increase the number of samples to 2 for two concurrent trials
                "resources_per_trial": {"gpu": 0.33},
            }

            sensibility_dir = f"{experiment_dir}/sensibility"
            for key, value in sensibility_dict.items():
                sensibility_dir_key = f"{sensibility_dir}/{key}"

                ray_experiments[sensibility_dir_key] = {
                    "run": TrainTransformer,
                    "config": {
                        **config_control,
                        key: value,
                        "exp_type": tune.grid_search([experiment]),
                        "positional_encoding": tune.grid_search([model]),
                    },
                    "local_dir": storage_path,
                    "stop": stop,
                    "checkpoint_config": air.CheckpointConfig(
                        checkpoint_frequency=CHECKPOINT_FREQUENCY,
                        checkpoint_at_end=True,
                    ),  # set num_to_keep=3 to save disk space
                    "num_samples": 1,  # Increase the number of samples to 2 for two concurrent trials
                    "resources_per_trial": {"gpu": 0.33},
                }

    return ray_experiments


tune.run_experiments(
    build_experiments(),
    resume="AUTO+ERRORED",
)
