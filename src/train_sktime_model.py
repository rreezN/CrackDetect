import torch
import wandb
import argparse
import os
import sys
import numpy as np

from tqdm import tqdm
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sktime.regression.base import BaseRegressor
from sktime.regression.dummy import DummyRegressor
from sktime.regression.kernel_based import RocketRegressor
from sktime.utils import mlflow_sktime

from src.util.utils import set_all_seeds
from src.data.dataloader import Platoon

import warnings
from sklearn.exceptions import DataConversionWarning
# Ignore specific warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='sktime.base._base_panel')

def create_batches(data, targets, batch_size):
    num_batches = len(data) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx], targets[start_idx:end_idx]

    if len(data) % batch_size != 0:
        start_idx = num_batches * batch_size
        yield data[start_idx:], targets[start_idx:]

def visualise_inhomo_array_object(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print(f"Row {i+1}, Column {j+1}: Length = {len(data[i, j])}")

def mse_score(model: BaseRegressor, data: np.ndarray, target: np.ndarray):
    from sklearn.metrics import mean_squared_error
    target_pred = model.predict(data)
    return -mean_squared_error(target, target_pred, sample_weight=None)


def train(model: BaseRegressor, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, batch_size: int = 1) -> None:

    iterator = tqdm(range(args.num_epochs), unit="epoch", position=0)
    best_model_loss = torch.inf

    for epoch in iterator:
        train_loss = []
        for data_segment, target_segment in train_loader:
            """ Batching and repeating """
            # for data, target in create_batches(data_segment, target_segment, batch_size):
            #     data_repeated = np.repeat(data[:, np.newaxis], target.shape[1], axis=1)
            #     model.fit(data_repeated, target)
            #     loss = model.score(data_repeated, target)
            #     train_loss.append(loss.item())

            for data, target in zip(data_segment, target_segment):
                # data_repeated = np.repeat(data[np.newaxis, :], len(target), axis=0)
                data = data[None, None, :]
                target = np.array([target])

                model.fit(data, target)
                # loss = model.score(data, target) # NOTE: R^2 does not work for single value target
                loss = mse_score(model, data, target)
                if isinstance(loss, float):
                    train_loss.append(loss)
                else:
                    train_loss.append(loss.item())

        val_loss = []
        for data_segment, target_segment in val_loader:
            """ Batching and repeating """
            # for data, target in create_batches(data_segment, target_segment, batch_size):
            #     data_repeated = np.repeat(data[:, np.newaxis], target.shape[1], axis=1)
            #     loss = model.score(data_repeated, target)
            #     val_loss.append(loss.item())
            
            for data, target in zip(data_segment, target_segment):
                data = data[None, :]
                target = target[None, :]

                model.fit(data, target)
                # loss = model.score(data, target) # NOTE: R^2 does not work for single value target
                loss = mse_score(model, data, target)
                if isinstance(loss, float):
                    val_loss.append(loss)
                else:
                    val_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)

        if val_loss < best_model_loss:
            best_model_loss = val_loss
            model_path = os.path.join("models", args.model_name, f"model_e{epoch}")
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            mlflow_sktime.save_model(sktime_model=model, path=model_path)

            # artifact = wandb.Artifact(name=f"model_e{epoch}", type='model')
            # artifact.add_dir(model_path)
            # run.log_artifact(artifact)

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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=10, help="Window size for data in meters")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    parser.add_argument("--only_iri", type=bool, default=True, help="Only use IRI data")

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
    trainset = Platoon(data_type='train', window_size=args.window_size, only_iri=args.only_iri)
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=1)
    valset = Platoon(data_type='val', window_size=args.window_size, only_iri=args.only_iri)
    val_loader = DataLoader(valset, batch_size=None, shuffle=False, num_workers=1)
    print("### Loading the data completed ###") if args.verbose else None
        
    # Define the model
    if args.model_name == "dummy_regressor":
        model = DummyRegressor(strategy="mean")
    elif args.model_name == "rocket_regressor":
        model = RocketRegressor(rocket_transform="multirocket")

    # Train the model
    print("### Training the model ###") if args.verbose else None
    train(model=model, train_loader=train_loader, val_loader=val_loader, args=args, batch_size=args.batch_size)
    print("### Training the model completed ###") if args.verbose else None

    run.finish()