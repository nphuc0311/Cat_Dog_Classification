from dataclasses import dataclass

@dataclass  # Simplifies the creation of a class to store parameters
class Args:
    cuda: str = "cuda:0"                                        # CUDA device for training (default: first available GPU)
    train_src: str = "/content/data/training_set/training_set"  # Path to the training dataset
    test_src: str = "/content/data/test_set/test_set"           # Path to the test dataset
    save_path: str = "/content/"                                # Directory to save model checkpoints or results
    seed: int = 42                                              # Random seed for reproducibility
    batch_size: int = 32                                        # Batch size for training
    num_epochs: int = 30                                        # Number of training epochs
    dropout: float = 0.1                                        # Dropout rate to prevent overfitting
    lr: float = 1e-3                                            # Learning rate for optimizer
    weight_decay: float = 1e-5                                  # Weight decay for regularization (L2 penalty)
    max_num_trials: int = 3                                     # Maximum number of hyperparameter optimization trials
