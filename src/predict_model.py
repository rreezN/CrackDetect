import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from matplotlib import pyplot as plt

from models.hydramr import HydraMR
from models.regressor import Regressor
from data.dataloader import Platoon
from torch.utils.data import DataLoader

def predict(model: torch.nn.Module, feature_extractpr: torch.nn.Module, testloader: torch.utils.data.DataLoader):
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.eval()
    
    predictions = torch.tensor([])
    targets = torch.tensor([])
    test_losses = np.array([])
    
    test_iterator = tqdm(testloader, unit="batch", position=0, leave=False)

    for data_segment, target_segment in test_iterator:
        for batch_data, batch_target in create_batches(data_segment, target_segment, batch_size=args.batch_size):
            batch_target = batch_target.to(torch.float32)
            batch_features = feature_extractor(batch_data)
            
            batch_mean = torch.mean(batch_features, dim=1).unsqueeze(1)
            batch_std = torch.std(batch_features, dim=1).unsqueeze(1)
            batch_features_normalised = (batch_features - batch_mean)/batch_std
            output = model(batch_features_normalised)
            
            loss_fn = nn.MSELoss()
            predictions = torch.cat((predictions, output), dim=0)
            targets = torch.cat((targets, batch_target), dim=0)
            
            test_loss = loss_fn(output, batch_target).item()
            test_losses = np.append(test_losses, test_loss)
            
            test_iterator.set_description(f'Overall MSE: {test_losses.mean():.2f} Batch MSE: {test_loss:.2f}')
            
            if args.plot_during:
                plot_predictions(predictions.detach().numpy(), targets.detach().numpy(), test_losses)
    
    return predictions.detach().numpy(), targets.detach().numpy(), test_losses

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


def plot_predictions(predictions, targets, test_losses, show=False):
    red_colors = ['lightcoral', 'firebrick', 'darkred', 'red']
    blue_colors = ['lightblue', 'royalblue', 'darkblue', 'blue']
    KPI = ['Damage Index (DI)', 'Rutting Index (RUT)', 'Patching Index (PI)', 'International Rougness Index (IRI)']
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes = axes.flat
    
    # Set title
    plt.suptitle(f'Predictions vs Targets, MSE: {np.mean(test_losses):.2f}', fontsize=24)
    
    # Plot each KPI
    for i in range(len(axes)):
        axes[i].title = axes[i].set_title(f'{KPI[i]}, MSE: {(np.mean(np.abs(targets[:, i] - predictions[:, i]))**2):.2f}')
        axes[i].plot(predictions[:, i], label="predicted", color='indianred', alpha=.75)
        axes[i].plot(targets[:, i], label="target", color='royalblue', alpha=.75)
        axes[i].legend()
    plt.tight_layout
    plt.savefig('reports/figures/model_results/predictions.pdf')
    if show:
        plt.show()
    plt.close()

def get_args():
    parser = ArgumentParser(description='Predict Model')
    parser.add_argument('--model', type=str, default='models/regressor_single_layer.pt')
    parser.add_argument('--data', type=str, default='data/processed/segments.hdf5".csv')
    parser.add_argument('--num_features', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--plot_during', action='store_false')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    feature_extractor = HydraMR(batch_size=args.batch_size, num_features=args.num_features)
    
    model = Regressor()
    model.load_state_dict(torch.load(args.model))
    
    # Load data
    testset = Platoon(data_type='test', pm_windowsize=1)
    test_loader = DataLoader(testset, batch_size=None, shuffle=False, num_workers=0)
    
    predictions, targets, test_losses = predict(model, feature_extractor, test_loader)
    
    plot_predictions(predictions, targets, test_losses)
    