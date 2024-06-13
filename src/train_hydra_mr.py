import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader 

from models.hydramr import HydraMRRegressor
from data.feature_dataloader import Features


def train(model: HydraMRRegressor, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int = 10, lr: float = 0.001):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set loss function
    loss_fn = nn.MSELoss()
    
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = np.inf
    
    # Iterate over segments of data (each segment is a time series where the minimum speed is above XX km/h)
    for i, epoch in enumerate(range(epochs)):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        model.train()
        train_losses = []
        for data, target in train_iterator:
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
        epoch_train_losses.append(np.mean(train_losses))
        
        val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
        model.eval()
        val_losses = []
        # bad_predictions = 0
        # bad_data = []
        # bad_max_idx = []
        # bad_min_idx = []
        for val_data, target in val_iterator:
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
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f'models/best_{model.name}.pt')
            print(f"Saving best model with mean val loss: {np.mean(val_losses):.3f} at epoch {epoch+1}")
            
        epoch_val_losses.append(np.mean(val_losses))

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
        plt.savefig(f'reports/figures/model_results/{model.name}/loss.pdf')
        plt.close()
    
    torch.save(model.state_dict(), f'models/{model.name}.pt')
    


def get_args():
    parser = ArgumentParser(description='Train the Hydra-MultiRocket model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (batches are concatenated MR and Hydra features).')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for the optimizer.')
    parser.add_argument('--feature_extractors', type=str, nargs='+', default=['MultiRocketMV_50000', 'HydraMV_8_64'], help='Feature extractors to use for prediction.')
    parser.add_argument('--name_identifier', type=str, default='', help='Name identifier for the feature extractors.')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation. Default is 5.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Define feature extractors
    # These are the names of the stored models/features (in features.hdf5)
    # e.g. ['MultiRocketMV_50000', 'HydraMV_8_64'] you can check the available features with check_hdf5.py
    feature_extractors = args.feature_extractors
    # feature_extractors = ['HydraMV_8_64']
    
    # If you have a name_identifier in the stored features, you need to include this in the dataset
    # e.g. to use features from "MultiRocketMV_50000_subset100," set name_identifier = "subset100"
    name_identifier = args.name_identifier
    
    for fold in range(args.folds):
        print(f"Training fold {fold+1}/{args.folds}")
        
        # Load data
        trainset = Features(data_type='train', feature_extractors=feature_extractors, name_identifier=name_identifier, fold=fold)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        valset = Features(data_type='val', feature_extractors=feature_extractors, name_identifier=name_identifier, fold=fold)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        input_shape, target_shape = trainset.get_data_shape()
        
        # Create model
        # As a baseline, MultiRocket_50000 will output 49728 features, Hydra_8_64 will output 5120 features, and there are 4 KPIs (targets)
        model = HydraMRRegressor(input_shape[0], target_shape[0], name=f'HydraMRRegressor_MultiRocketMV_50000_HydraMV_8_64_50_latent_dim') 
        
        # Train
        # TODO: Modify train function to return best_model, best_val_loss, and training curves
        train(model, train_loader, val_loader)
    