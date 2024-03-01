import torch
import wandb
import sktime
import argparse

DEFAULT_PARAMS = {
    "model_name": "model",
    "project_name": "Mini-model investigation",
    "seed": 11,
    "num_epochs": 150,
    "patience": 30,
    "batch_dict": {0: 8,
                   4: 16,
                   8: 24,
                   14: 32,
                   20: 48,
                   28: 64,
                   36: 96,
                   48: 128},
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "limit_train_batches": 1.0,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "activation_function": "LeakyReLU"
}

def train():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training options")
    parser.add_argument("--model_name", type=str, default="model")
    # parser.add_argument("--project_name", type=str, default="audiobots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--window_size", type=int, default=10, help="Window size for data in meters")

    # TODO: Set seed
    train()