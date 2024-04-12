import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from matplotlib import pyplot as plt

from models.hydramr import HydraMRRegressor
from data.feature_dataloader import Features
from torch.utils.data import DataLoader

def predict(model: torch.nn.Module, testloader: torch.utils.data.DataLoader):
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.eval()
    
    all_predictions = torch.tensor([])
    all_targets = torch.tensor([])
    test_losses = np.array([])
    
    test_iterator = tqdm(testloader, unit="batch", position=0, leave=False)

    kpi_maxs = torch.tensor(testset.kpi_maxs)
    kpi_mins = torch.tensor(testset.kpi_mins)
    
    for data, targets in test_iterator:
        output = model(data)
        
        # Transform targets and predictions to original scale from [0, 1]
        output = (((output - 0) * (kpi_maxs - kpi_mins)) / (1 - 0)) + kpi_mins
        targets = (((targets - 0) * (kpi_maxs - kpi_mins)) / (1 - 0)) + kpi_mins
        
        loss_fn = nn.MSELoss()
        all_predictions = torch.cat((all_predictions, output), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)
        
        test_loss = torch.sqrt(loss_fn(output, targets)).item()
        test_losses = np.append(test_losses, test_loss)
        
        test_iterator.set_description(f'Overall RMSE: {test_losses.mean():.2f} Batch RMSE: {test_loss:.2f}')
        
        if args.plot_during:
            plot_predictions(all_predictions.detach().numpy(), all_targets.detach().numpy(), test_losses)

    all_predictions = all_predictions.detach().numpy()
    all_targets = all_targets.detach().numpy()
    plot_predictions(all_predictions, all_targets, test_losses)
        
    return all_predictions, all_targets, test_losses


def plot_predictions(predictions, targets, test_losses, show=False):
    red_colors = ['lightcoral', 'firebrick', 'darkred', 'red']
    blue_colors = ['lightblue', 'royalblue', 'darkblue', 'blue']
    KPI = ['Damage Index (DI)', 'Rutting Index (RUT)', 'Patching Index (PI)', 'International Rougness Index (IRI)']
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes = axes.flat
    
    # Set title
    plt.suptitle(f'Predictions vs Targets, RMSE: {np.mean(test_losses):.2f}', fontsize=24)
    
    # Plot each KPIs
    for i in range(len(axes)):
        axes[i].title = axes[i].set_title(f'{KPI[i]}, RMSE: {(np.sqrt(np.mean(np.abs(targets[:, i] - predictions[:, i]))**2)):.2f}')
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
    parser.add_argument('--model', type=str, default='models/best_HydraMRRegressor.pt')
    parser.add_argument('--data', type=str, default='data/processed/features.hdf5".csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--plot_during', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    model = HydraMRRegressor()
    model.load_state_dict(torch.load(args.model))
    
    # Load data
    testset = Features(data_type='test')
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    predictions, targets, test_losses = predict(model, test_loader)
    
    plot_predictions(predictions, targets, test_losses)
    