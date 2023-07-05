import os
from parameters import EPOCHS, EPOCH_N_METRICS

from trainable.trainable import TrainTransformer
from ray import air, tune

# Installation with !pip install -q -U ray[tune]


#### Hyperparameter Search with fault tolerance ####

# Set serialisation path
# storage_path = "/content/drive/MyDrive/ray_results" # Using Google Drive/Colab
storage_path = "./runs"
experiment = "grid_search"
stop = {  # number of epochs (it should include both ends)
    "training_iteration": EPOCHS + 1
}

config = {
    "learning_rate": tune.grid_search([1e-4]),  # [1e-5, 1e-4, 1e-3]
    "d_model": tune.grid_search([32, 16]),  # [16, 32, 64] ; Add 128?
    "n_heads": tune.grid_search([2]),  # [1, 2, 4]
    "dropout": tune.grid_search([0.0]),  # [0.0, 0.3] ; Add 0.5?
    "gradient_clipping": tune.grid_search([0]),  # [0, 0.5]
    "exp_type": tune.grid_search(  # ["iid", "ood"] or ["toy"] for testing the search
        ["toy"]
    ),
    "positional_encoding": tune.grid_search(["original"]),  # ["original", "embedding"]
}

tune.run(
    TrainTransformer,
    config=config,
    local_dir=storage_path,
    name=experiment,
    resume="AUTO+ERRORED",  # Enables fault tolerance
    num_samples=1,
    search_alg=None,
    stop=stop,  # number of epochs (it should include both ends)
    checkpoint_freq=EPOCH_N_METRICS,
    verbose=1,
)
