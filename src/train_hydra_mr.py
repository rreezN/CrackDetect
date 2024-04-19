import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader 

from models.hydramr import HydraMRRegressor
from models.regressor import Regressor
from data.feature_dataloader import Features


def train(model, train_loader, val_loader, epochs=100):
    # val_iterator = tqdm(val_loader, unit="batch", position=1, leave=False)
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
                train_iterator.set_description(f"Training Epoch {epoch}/{epochs}, Train loss: {loss.item():.3f}, Last epoch train loss: {epoch_train_losses[i-1]:.3f}, Last epoch val loss: {epoch_val_losses[i-1]:.3f}")
            else:
                train_iterator.set_description(f"Training Epoch {epoch}/{epochs}, Train loss: {loss.item():.3f}")
        epoch_train_losses.append(np.mean(train_losses))
        
        val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
        model.eval()
        val_losses = []
        for data, target in val_iterator:
            output = model(data)
            val_loss = loss_fn(output, target)
            val_losses.append(val_loss.item())
            val_iterator.set_description(f"Validating Epoch {epoch}/{epochs}, Train loss: {loss.item():.3f}, Last epoch train loss: {epoch_train_losses[i-1]:.3f}, Val loss: {val_loss:.3f}, Mean Val Loss: {np.mean(val_losses):.3f}")
        
        # save best model
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), f'models/best_{model.name}.pt')
            print(f"Saving best model with mean val loss: {np.mean(val_losses):.3f} at epoch {epoch}")
            
        epoch_val_losses.append(np.mean(val_losses))
    
        plt.plot(epoch_train_losses, label="Train loss")
        plt.plot(epoch_val_losses, label="Val loss")
        plt.title('Loss per epoch')
        plt.ylim(0, 1)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'reports/figures/model_results/{model.name}_loss.pdf')
        plt.close()
    
    torch.save(model.state_dict(), f'models/{model.name}.pt')
    


def get_args():
    parser = ArgumentParser(description='Hydra-MR')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--regressor_batch_size', type=int, default=8)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Load data
    trainset = Features(data_type='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    valset = Features(data_type='val')
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Create model
    model = HydraMRRegressor(49728+5120, 4) # Hardcoded for now, 49728=Features from MultiRocket, 5120=Features from Hydra
    
    # Train
    train(model, train_loader, val_loader, args.epochs)
    