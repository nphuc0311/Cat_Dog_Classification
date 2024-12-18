from dataclasses import dataclass

@dataclass                    # To simplify the creation of classes that are mainly used to store data.
class Args:
    cuda: str = "cuda:0"
    train_src: str = "/content/data/training_set/training_set"
    val_src: str = "/content/data/val_set/val_set"
    test_src: str = "data/test_set/test_set"
    save_path: str = "/content/"
    seed: int = 39
    batch_size: int = 32
    num_epochs: int = 30
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_num_trials: int = 5                             # Maximum number of hyperparameter optimization trials
