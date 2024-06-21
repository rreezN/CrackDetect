import os
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader

from data.feature_dataloader import Features
from models.hydramr import HydraMRRegressor, HydraMRRegressor_old

def predict(model: torch.nn.Module, testloader: torch.utils.data.DataLoader, path_to_model: str = None, plot_during: bool = False):
    """Run prediction for a given model and dataloader.
    
    Parameters:
    ----------
        model: model to use for prediction
        testloader: dataloader to use for prediction
    
    Returns:
    -------
        all_predictions (torch.Tensor), all_targets (torch.Tensor), test_losses (ndarray): predictions, targets and losses for the test set

    """
    model.eval()
    
    all_predictions = torch.tensor([])
    all_targets = torch.tensor([])
    test_losses = np.array([])
    
    test_iterator = tqdm(testloader, unit="batch", position=0, leave=False)
    kpi_means = torch.tensor(testloader.dataset.kpi_means)
    kpi_stds = torch.tensor(testloader.dataset.kpi_stds)
    
    for data, targets in test_iterator:
        output = model(data)
        
        # Convert back from standardized to original scale
        output = ((output * kpi_stds) + kpi_means)
        targets = ((targets * kpi_stds) + kpi_means)
        
        loss_fn = nn.MSELoss()
        all_predictions = torch.cat((all_predictions, output), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)
        
        test_loss = loss_fn(output, targets).item()
        test_losses = np.append(test_losses, test_loss)
        
        test_iterator.set_description(f'Overall RMSE (loss): {np.sqrt(test_losses.mean()):.2f} Batch RMSE (loss): {np.sqrt(test_loss):.2f}')
        
        if plot_during:
            plot_predictions(all_predictions.detach().numpy(), all_targets.detach().numpy(), show=True, args=args, path_to_model=path_to_model)
        
    return all_predictions, all_targets, test_losses


def calculate_errors(predictions, targets):
    """Calculate the errors between the predictions and the targets.
    
    Parameters:
    ----------
        predictions (np.array): The predictions made by the model.
        targets (np.array): The true values of the targets.
    
    Returns:
    -------
        RMSE (np.array): The Root Mean Squared Error between the predictions and the targets for each KPI.
        baseline_RMSE (np.array): The Root Mean Squared Error between the targets and the mean of the targets for each KPI.
    """
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    
    # Threshold the predictions to be minimum 0
    predictions = np.maximum(predictions, 0)
    
    # Calculate RMSE between targets and predictions for each KPI
    RMSE = np.sqrt(np.mean(np.abs(targets - predictions)**2, axis=0))
    
    # Calculate baseline RMSE between targets and predictions for each KPI
    baseline_RMSE = np.sqrt(np.mean(np.abs(targets - np.mean(targets, axis=0))**2, axis=0))
    
    return RMSE, baseline_RMSE

def plot_predictions(predictions, targets, show: bool = False, save: bool = True, args: Namespace = None, path_to_model: str = None):
    """Plot the predictions against teh targets in a scatter plot. Also reports the RMSE and correlation between the predictions and the targets.

    Args:
        predictions (torch.Tensor): Targets predicted by the model.
        targets (torch.Tensor): Real targets.
        show (bool, optional): Whether or not to show the plot in addition to saving it. Defaults to False.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    
    # Threshold the predictions to be minimum 0
    predictions = np.maximum(predictions, 0)
    
    red_colors = ['lightcoral', 'firebrick', 'darkred', 'red']
    blue_colors = ['lightblue', 'royalblue', 'darkblue', 'blue']
    KPI = ['Damage Index (DI)', 'Rutting Index (RUT)', 'Patching Index (PI)', 'International Rougness Index (IRI)']
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes = axes.flat
    
    # Get errors
    rmse, baseline_rmse = calculate_errors(predictions, targets)
    
    correlations = []
    # Plot each KPIs
    for i in range(len(axes)):
        correlation = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
        correlations.append(correlation)
        
        # Set limits
        xlim_min = min(min(targets[:, i]), min(predictions[:, i])) - 0.1
        ylim_min = min(min(targets[:, i]), min(predictions[:, i])) - 0.1
        xlim_max = max(max(targets[:, i]), max(predictions[:, i])) + 0.1
        ylim_max = max(max(targets[:, i]), max(predictions[:, i])) + 0.1
        axes[i].set_xlim(xlim_min, xlim_max)
        axes[i].set_ylim(ylim_min, ylim_max)
        
        # Plot diagonal
        axes[i].plot([xlim_min, xlim_max], [ylim_min, ylim_max], color='black', linestyle='dashed', linewidth=1, alpha=.75, zorder=1)
        
        # Plot mean target
        axes[i].axhline(np.mean(targets[:, i]), color='goldenrod', linestyle='dashed', linewidth=1, alpha=.75, label='Mean Target (baseline)', zorder=1)
        
        # Plot scatter
        axes[i].scatter(targets[:, i], predictions[:, i], color='indianred', alpha=.75, label='Predicted vs Target', zorder=0)
        # axes[i].scatter(targets[:, i], np.full(len(targets), np.mean(targets[:, i])), color='goldenrod', alpha=.75, label='Mean Target (baseline)')
        axes[i].set_xlabel('Target', fontsize=12)
        axes[i].set_ylabel('Prediction', fontsize=12)
        
        # Set title
        axes[i].title = axes[i].set_title(f'{KPI[i]}\nRMSE: {rmse[i]:.2f}, baseline RMSE: {baseline_rmse[i]:.2f}, correlation: {correlation:.2f}')
        axes[i].legend()
    
    data_type_name = args.data_type.capitalize()
    # loss: {np.mean(test_losses):.2f},
    plt.suptitle(f'{data_type_name} Predictions vs Targets\nRMSE: {np.mean(rmse):.2f}, baseline RMSE: {np.mean(baseline_rmse):.2f}. correlation: {np.mean(correlations):.2f}', fontsize=24)
    plt.tight_layout()
    if save:
        plt.savefig(f'reports/figures/model_results/{path_to_model}/{args.data_type}_predictions.pdf')
    if show:
        plt.show()
    plt.close()
    

def get_args(external_parser: ArgumentParser = None):
    if external_parser is None:
        parser = ArgumentParser(description='Predict Model')
    else:
        parser = external_parser
    parser.add_argument('--model', type=str, default='models/best_HydraMRRegressor.pt', help='Path to the model file.')
    parser.add_argument('--data', type=str, default='data/processed/features.hdf5', help='Path to the data file.')
    parser.add_argument('--feature_extractors', type=str, nargs='+', default=['HydraMV_8_64'], help='Feature extractors to use for prediction.')
    parser.add_argument('--name_identifier', type=str, default='', help='Name identifier for the feature extractors.')
    parser.add_argument('--data_type', type=str, default='test', help='Type of data to use for prediction.', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction.')
    parser.add_argument('--plot_during', action='store_true', help='Plot predictions during prediction.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the model.')
    parser.add_argument('--fold', type=int, default=1, help='Fold to use for prediction.')
    parser.add_argument('--model_depth', type=int, default=0, help='Number of hidden layers in the model. Default is 0')
    parser.add_argument('--batch_norm', action="store_true", help='If batch normalization is used in the model')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file. Default is False')
    
    if external_parser is None:
        return parser.parse_args()
    else:
        return parser

def main(args: Namespace):
    # Load data
    testset = Features(args.data, data_type=args.data_type, feature_extractors=args.feature_extractors, name_identifier=args.name_identifier, fold=args.fold)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_shape, target_shape = testset.get_data_shape()
    
    model = HydraMRRegressor(in_features=input_shape[0], out_features=target_shape[0], hidden_dim=args.hidden_dim, model_depth=args.model_depth, batch_norm=args.batch_norm)  # MultiRocket+Hydra 49728+5120
    model.load_state_dict(torch.load(args.model))
    
    path_to_model = Path(args.model).stem
    if path_to_model.startswith('best_'):
        path_to_model = path_to_model[5:]
    os.makedirs(f'reports/figures/model_results/{path_to_model}', exist_ok=True)
    
    predictions, targets, test_losses = predict(model, test_loader, plot_during=args.plot_during, path_to_model=path_to_model)
    if args.save_predictions:
        np.save(f'reports/figures/model_results/{path_to_model}/{args.data_type}_predictions.npy', predictions.detach().numpy())
        np.save(f'reports/figures/model_results/{path_to_model}/{args.data_type}_targets.npy', targets.detach().numpy())

    plot_predictions(predictions, targets, show=False, args=args, path_to_model=path_to_model)


if __name__ == '__main__':
    args = get_args()
    main(args)
    
    
    