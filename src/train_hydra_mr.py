import os
import torch
import wandb
import numpy as np
import torch.nn as nn
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader 

from models.hydramr import HydraMRRegressor
from data.feature_dataloader import Features
from util.utils import set_all_seeds

# Set seed for reproducibility
set_all_seeds(42)


def train(model: HydraMRRegressor, 
          train_loader: DataLoader, 
          val_loader: DataLoader,
          fold: int,
          epochs: int = 10, 
          lr: float = 0.001,
          ):
    """Training loop for the Hydra-MR model.
    
    This function trains the Hydra-MR model using the provided training data and validation data loaders.
    The model is trained for a specified number of epochs with a given learning rate.

    The model is saved in the models directory with the name of the model along with the best model during training.
    Training curves are saved in the reports/figures/model_results directory.
    
    Parameters:
    ----------
        model (HydraMRRegressor): The model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
        lr (float, optional): Learning rate of the optimizer. Defaults to 0.001.
    """
    
    if args.lr != 0.001:
        lr = args.lr
    if args.epochs != 10:
        epochs = args.epochs
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    # Set loss function
    loss_fn = nn.MSELoss()

    wandb.config.update({"optimizer": str(optimizer), "loss_function": str(loss_fn)})

    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = np.inf
    
    # Iterate over segments of data (each segment is a time series where the minimum speed is above XX km/h)
    for i, epoch in enumerate(range(epochs)):
        start = time.time()
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        model.train()
        train_losses = []
        for data, target in train_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = loss_fn(output, target)
            train_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if len(epoch_val_losses) > 0:
                train_iterator.set_description(f"Training Epoch {epoch+1}/{epochs}, Train loss: {loss.item():.3f}, Last epoch train loss: {epoch_train_losses[i-1]:.3f}, Last epoch val loss: {epoch_val_losses[i-1]:.3f}")
            else:
                train_iterator.set_description(f"Training Epoch {epoch+1}/{epochs}, Train loss: {loss.item():.3f}")

        mean_train_loss = np.mean(train_losses)
        wandb.log({f"epoch_{fold}": epoch+1, f"train_loss_{fold}": mean_train_loss})
        epoch_train_losses.append(mean_train_loss)
        
        val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
        model.eval()
        val_losses = []
        # bad_predictions = 0
        # bad_data = []
        # bad_max_idx = []
        # bad_min_idx = []
        for val_data, target in val_iterator:
            val_data, target = val_data.to(device), target.to(device)
            output = model(val_data)
            val_loss = loss_fn(output, target)

            # TODO: Fix insane predictions in validation, and then switch to MSE
            # Validation losses explodeeeeee
            if val_loss > 100:
                # bad_predictions += 1
                # bad_data.append([val_data[val_data > 10], val_data[val_data < -10]])
                # bad_max_idx.append((val_data == torch.max(val_data)).nonzero())
                # bad_min_idx.append((val_data == torch.min(val_data)).nonzero())
                print(f"Val loss: {val_loss}")
            val_losses.append(val_loss.item())
            val_iterator.set_description(f"Validating Epoch {epoch+1}/{epochs}, Train loss: {loss.item():.3f}, Last epoch train loss: {epoch_train_losses[i-1]:.3f}, Val loss: {val_loss:.3f}, Mean Val Loss: {np.mean(val_losses):.3f}")
        
        # save best model
        if np.mean(val_losses) < best_val_loss:
            end = time.time()
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f'models/best_{model.name}_{fold}.pt')
            print(f"Saving best model with mean val loss: {np.mean(val_losses):.3f} at epoch {epoch+1} ({end-start:.2f}s)")
            # Note to windows users: you may need to run the script as administrator to save the model
            wandb.save(f'models/best_{model.name}_{fold}.pt')
            
        mean_val_loss = np.mean(val_losses)
        wandb.log({f"epoch_{fold}": epoch+1, f"val_loss_ {fold}": mean_val_loss})
        epoch_val_losses.append(mean_val_loss)

        x = np.arange(1, epoch+2, step=1)
        plt.plot(x, epoch_train_losses, label="Train loss")
        plt.plot(x, epoch_val_losses, label="Val loss")
        # plt.xticks(x)
        plt.title('Loss per epoch')
        plt.ylim(0, min(5, max(max(epoch_train_losses), max(epoch_val_losses))+1))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        os.makedirs(f'reports/figures/model_results/{model.name}', exist_ok=True)
        plt.savefig(f'reports/figures/model_results/{model.name}/loss_{fold}.pdf')
        plt.close()
    
    torch.save(model.state_dict(), f'models/{model.name}.pt')

    return epoch_train_losses, epoch_val_losses, best_val_loss
    

def get_args():
    parser = ArgumentParser(description='Train the Hydra-MultiRocket model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train. Default 10')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (batches are concatenated MR and Hydra features). Default 32')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for the optimizer. Default 1e-6')
    parser.add_argument('--feature_extractors', type=str, nargs='+', default=['HydraMV_8_64'], help='Feature extractors to use for prediction. Default is MultiRocketMV_50000 and HydraMV_8_64.')
    parser.add_argument('--name_identifier', type=str, default='', help='Name identifier for the feature extractors. Default is empty.')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation. Default is 5.')
    parser.add_argument('--model_name', type=str, default='HydraMRRegressor', help='Name of the model. Default is HydraMRRegressor.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer. Default is 0.0')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the model. Default is 64')
    parser.add_argument('--project_name', type=str, default='hydra_mr_test', help='Name of the project on wandb. Default is hydra_mr_test to ensure we do not write into something important.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the model. Default is 0.5')
    parser.add_argument('--model_depth', type=int, default=0, help='Number of hidden layers in the model. Default is 1')
    parser.add_argument('--batch_norm', type=bool, default=True, help='Whether to use batch normalization in the model. Default is False')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}.")
 

    wandb.init(project=args.project_name, entity='fleetyeet')
    wandb.config.update(args)
    # Define feature extractors
    # These are the names of the stored models/features (in features.hdf5)
    # e.g. ['MultiRocketMV_50000', 'HydraMV_8_64'] you can check the available features with check_hdf5.py
    feature_extractors = args.feature_extractors
    
    # If you have a name_identifier in the stored features, you need to include this in the dataset
    # e.g. to use features from "MultiRocketMV_50000_subset100," set name_identifier = "subset100"
    name_identifier = args.name_identifier

    train_losses = []
    val_losses = []
    best_val_losses = []
    
    for fold in range(args.folds):
        print(f"Training fold {fold+1}/{args.folds}")
        
        # Load data
        if args.feature_extractors == ['MultiRocketMV_50000+HydraMV_8_64']:
            trainset = Features(data_type='train', feature_extractors=['MultiRocketMV_50000', 'HydraMV_8_64'], name_identifier=name_identifier, fold=fold, verbose=False)
            valset = Features(data_type='val', feature_extractors=['MultiRocketMV_50000', 'HydraMV_8_64'], name_identifier=name_identifier, fold=fold, verbose=False)
        else:
            trainset = Features(data_type='train', feature_extractors=feature_extractors, name_identifier=name_identifier, fold=fold, verbose=False)
            valset = Features(data_type='val', feature_extractors=feature_extractors, name_identifier=name_identifier, fold=fold, verbose=False)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        input_shape, target_shape = trainset.get_data_shape()
        
        # Create model
        # As a baseline, MultiRocket_50000 will output 49728 features, Hydra_8_64 will output 5120 features, and there are 4 KPIs (targets)
        model = HydraMRRegressor(in_features=input_shape[0], 
                                 out_features=target_shape[0], 
                                 hidden_dim=args.hidden_dim, 
                                 dropout=args.dropout,
                                 name=args.model_name,
                                 model_depth=args.model_depth,
                                 batch_norm=args.batch_norm
                                 ).to(device)
        
        wandb.watch(model, log='all')
        wandb.config.update({f"model_{fold}": model.name})

        # Train
        k_fold_train_losses, k_fold_val_losses, best_val_loss = train(model, train_loader, val_loader, fold=fold)
    
        train_losses.append(k_fold_train_losses)
        val_losses.append(k_fold_val_losses)
        best_val_losses.append(best_val_loss)
    
    mean_best_val_loss = np.mean(best_val_losses)
    std_best_val_loss = np.std(best_val_losses)
    print(f"Mean best validation loss: {mean_best_val_loss:.3f}")
    print(f"Standard deviation of best validation loss: {std_best_val_loss:.3f}")
    wandb.log({"mean_best_val_loss": mean_best_val_loss, "std_best_val_loss": std_best_val_loss})

    # plot training curves
    for i in range(args.folds):
        x = np.arange(1, args.epochs+1, step=1)
        plt.plot(x, train_losses[i], c="b", alpha=0.2)
        plt.plot(x, val_losses[i], linestyle='--', c="r", alpha=0.2)

    plt.plot(x, np.mean(train_losses, axis=0), label="Train loss", c="b")
    plt.plot(x, np.mean(val_losses, axis=0), label="Val loss", c="r", linestyle='--')
    plt.ylim(0, min(7, max(np.max(train_losses), np.max(val_losses))+0.2))
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs(f'reports/figures/model_results/{model.name}', exist_ok=True)
    plt.savefig(f'reports/figures/model_results/{model.name}/loss_combined.pdf')
    # log the plot to wandb
    wandb.log({"loss_combined": wandb.Image(plt)})
    plt.close()

    wandb.finish()