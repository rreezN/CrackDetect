import torch
import wandb
import argparse
import os
import sys

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sktime.regression.dummy import DummyRegressor
from sktime.utils import mlflow_sktime

from src.util.utils import set_all_seeds
from src.data.dataloader import Platoon

def train(model: DummyRegressor, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace) -> None:

    iterator = tqdm(range(args.num_epochs), unit="epoch", position=0)
    best_model_loss = torch.inf

    for epoch in iterator:
        train_loss = []
        for data, target in train_loader:
            model.fit(data, target)
            loss = model.score(data, target)
            train_loss.append(loss.item())

        val_loss = []
        for data, target in val_loader:
            loss = model.score(data, target)
            val_loss.append(loss.item())
        
        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)

        if val_loss < best_model_loss:
            best_model_loss = val_loss
            model_path = os.path.join("models", args.model, f"model_e{epoch}.pt")
            mlflow_sktime.save_model(sktime_model=model, path=model_path)

            artifact = wandb.Artifact(name=f"model_e{epoch}", type='model')
            artifact.add_file(model_path)
            run.log_artifact(artifact)

        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "epoch": epoch
            })
        
        iterator.set_description(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training options")
    parser.add_argument("--model_name", type=str, default="dummy_regressor")
    parser.add_argument("--project_name", type=str, default="Version 0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=10, help="Window size for data in meters")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    args = parser.parse_args()
    print(args) if args.verbose else None

    # Set all seed
    set_all_seeds(args.seed)

    run = wandb.init(
        project=args.project_name,
        entity="fleetyeet",
        # group=group,
        # Track hyperparameters and run metadata
        config=args
    )

    # Load the data
    print("### Loading the data ###") if args.verbose else None
    trainset = Platoon(data_type='train', window_size=args.window_size)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    valset = Platoon(data_type='val', window_size=args.window_size)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=3)
    print("### Loading the data completed ###") if args.verbose else None


    # Define the model
    if args.model_name == "dummy_regressor":
        model = DummyRegressor(strategy="mean")

    # Train the model
    print("### Training the model ###") if args.verbose else None
    train(model=model, train_loader=train_loader, val_loader=val_loader, args=args)
    print("### Training the model completed ###") if args.verbose else None

    run.finish()