import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from models.hydramr import HydraMR
from models.regressor import Regressor
from data.dataloader import Platoon
from torch.utils.data import DataLoader 
from tqdm import tqdm
from matplotlib import pyplot as plt


def create_batches(data, targets, batch_size):
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx], targets[start_idx:end_idx]

    """
    NOTE - this is a generator, so the last batch will be smaller, but for certain models 
    we have to have fixed batch size so we can't just yield the last batch (thus, at times we will lose some data)
    """    
    if len(data) % batch_size != 0:
        start_idx = num_batches * batch_size
        yield data[start_idx:], targets[start_idx:]


def train(feature_extractor, regressor, train_loader, test_loader, epochs=10, batch_size=10, regressor_batch_size=64):
    train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
    
    # Set optimizer
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)
    # Set loss function
    loss_fn = nn.MSELoss()
    
    regressor.train()
    losses = []
    # Iterate over segments of data (each segment is a time series where the minimum speed is above XX km/h)
    for data_segment, target_segment in train_iterator:
        # Split the segment into batches (batch_size is in seconds)
        for batch_data, batch_target in create_batches(data_segment, target_segment, batch_size=batch_size):
            batch_target = batch_target.to(torch.float32)
            batch_features = feature_extractor(batch_data)
            
            # Create dataset over features
            # feature_dataset = torch.utils.data.TensorDataset(batch_features, batch_target)
            # feature_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=regressor_batch_size, shuffle=True)
            
            # Train regressor
            
            # regressor_iterator = tqdm(feature_dataloader, unit="i", position=1, leave=False)
            # for data in regressor_iterator:
            #     mini_batch_features, mini_batch_target = data
            #     mini_batch_target = mini_batch_target.to(torch.float32)
            batch_mean = torch.mean(batch_features, dim=1).unsqueeze(1)
            batch_std = torch.std(batch_features, dim=1).unsqueeze(1)
            batch_features_normalised = (batch_features - batch_mean)/batch_std
            output = regressor(batch_features_normalised)
            # print("\n")
            # print(output,"\n", batch_target)
            loss = loss_fn(output, batch_target)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_iterator.set_description(f"Loss: {loss.item():.3f}")
            
            # Print loss
            # print(f"Loss: {loss.item()}")
            # regressor_iterator.set_description(f"Loss: {loss.item()}")   
    
    torch.save(regressor.state_dict(), 'models/regressor.pt')
    
    plt.plot(losses)
    plt.title('Loss')
    plt.savefig('reports/figures/model_results/loss.pdf')


def get_args():
    parser = ArgumentParser(description='Hydra-MR')
    parser.add_argument('--num_features', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--regressor_batch_size', type=int, default=8)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Load data
    trainset = Platoon(data_type='train', pm_windowsize=1)
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=0)
    
    testset = Platoon(data_type='val', pm_windowsize=1)
    test_loader = DataLoader(testset, batch_size=None, shuffle=True, num_workers=0)
    
    # Create model
    feature_extractor = HydraMR(batch_size=args.batch_size, num_features=args.num_features)
    regressor = Regressor(49728+5120, 4) # Hardcoded for now, 49728=Features from MultiRocket, 5120=Features from Hydra
    
    # Train
    train(feature_extractor, regressor, train_loader, test_loader, batch_size=args.batch_size)
    