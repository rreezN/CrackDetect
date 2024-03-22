import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from models.hydramr import HydraMR
from data.dataloader import Platoon
from torch.utils.data import DataLoader 
from tqdm import tqdm


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


def train_model(feature_extractor, regressor, train_loader, test_loader, epochs=10, batch_size=10):
    train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
    for data_segment, target_segment in train_iterator:
        for data, target in create_batches(data_segment, target_segment, batch_size=batch_size):
            features = feature_extractor(data)
            print(data.shape, target.shape)

07


def get_args():
    parser = ArgumentParser(description='Hydra-MR')
    parser.add_argument('--num_features', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=100)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Load data
    trainset = Platoon(data_type='train', pm_windowsize=2)
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=0)
    
    testset = Platoon(data_type='test', pm_windowsize=2)
    test_loader = DataLoader(testset, batch_size=None, shuffle=True, num_workers=0)
    
    # Create model
    feature_extractor = HydraMR(args.num_features)
    regressor = nn.Sequential(nn.Linear(10000, 4))
    
    # Train
    train_model(feature_extractor, regressor, train_loader, test_loader)
    
    
    print("yo")