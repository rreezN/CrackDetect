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
    
    Parameters:
    ----------
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns:
    -------
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.eval()
    
    all_predictions = torch.tensor([])
    all_targets = torch.tensor([])
    test_losses = np.array([])
    
    test_iterator = tqdm(testloader, unit="batch", position=0, leave=False)

    kpi_means = torch.tensor(testset.kpi_means)
    kpi_stds = torch.tensor(testset.kpi_stds)
    
    for data, targets in test_iterator:
        output = model(data)
        
        # Convert back from standardized to original scale
        output = ((output * kpi_stds) + kpi_means)
        targets = ((targets * kpi_stds) + kpi_means)
        
        loss_fn = nn.MSELoss()
        all_predictions = torch.cat((all_predictions, output), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)
        
        test_loss = torch.sqrt(loss_fn(output, targets)).item()
        test_losses = np.append(test_losses, test_loss)
        
        test_iterator.set_description(f'Overall RMSE (loss): {test_losses.mean():.2f} Batch RMSE (loss): {test_loss:.2f}')
        
        if args.plot_during:
            plot_predictions(all_predictions.detach().numpy(), all_targets.detach().numpy(), test_losses)
        
    return all_predictions, all_targets, test_losses


def calculate_errors(predictions, targets, lags: int = 10):
    """Calculate the errors between the predictions and the targets.
    
    Parameters:
    ----------
        predictions (np.array): The predictions made by the model.
        targets (np.array): The true values of the targets.
    
    Returns:
    -------
        RMSE (np.array): The Root Mean Squared Error between the predictions and the targets for each KPI.
        baseline_RMSE (np.array): The Root Mean Squared Error between the targets and the mean of the targets for each KPI.
        best_correlation (tuple): The best correlation between the predictions and the targets for each KPI.
    """
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    
    # Calculate RMSE between targets and predictions for each KPI
    RMSE = np.sqrt(np.mean(np.abs(targets - predictions)**2, axis=0))
    
    # Calculate baseline RMSE between targets and predictions for each KPI
    baseline_RMSE = np.sqrt(np.mean(np.abs(targets - np.mean(targets, axis=0))**2, axis=0))
    
    # Calculate correlation between targets and predictions for each KPI with different lags
    correlations = []
    for lag in range(0, lags):
        for i in range(targets.shape[1]):
            if lag == 0:
                correlation = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
            else:
                correlation = np.corrcoef(targets[:-lag, i], predictions[lag:, i])[0, 1]
                
            correlations.append(correlation)
    
    # Save best correlation for each target
    best_correlations = np.zeros(targets.shape[1])
    best_lags = np.zeros(targets.shape[1])
    
    # find the best correlation and lag for each KPI
    for i in range(targets.shape[1]):
        best_correlations[i] = max(correlations[i::lags])
        best_lags[i] = np.argmax(correlations[i::lags])
    
    return RMSE, baseline_RMSE, (best_lags, best_correlations)


def plot_predictions(predictions: torch.Tensor, targets: torch.Tensor, test_losses: np.ndarray, show: bool = False):
    """Plot the predictions against the targets. Also reports the RMSE and correlation between the predictions and the targets.

    Args:
        predictions (_type_): _description_
        targets (_type_): _description_
        test_losses (_type_): _description_
        show (bool, optional): _description_. Defaults to False.
    """
    
    predictions = predictions.detach().numpy()
    targets = targets.detach().numpy()
    
    red_colors = ['lightcoral', 'firebrick', 'darkred', 'red']
    blue_colors = ['lightblue', 'royalblue', 'darkblue', 'blue']
    KPI = ['Damage Index (DI)', 'Rutting Index (RUT)', 'Patching Index (PI)', 'International Rougness Index (IRI)']
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes = axes.flat
    
    # Get errors
    # TODO: Make sure this works... Might need to change the way errors are returned
    # Or the way they are plotted
    rmse, baseline_rmse, correlation = calculate_errors(predictions, targets)
    correlations = correlation[1]
    lags = correlation[0]
    
    # Plot each KPIs
    for i in range(len(axes)):
        axes[i].title = axes[i].set_title(f'{KPI[i]}, RMSE: {rmse[i]:.2f}, correlation: {correlations[i]:.2f}, baseline RMSE: {baseline_rmse[i]:.2f}')
        axes[i].plot(predictions[:, i], label="predicted", color='indianred', alpha=.75)
        axes[i].plot(targets[:, i], label="target", color='royalblue', alpha=.75)
        axes[i].plot(np.full(len(targets), np.mean(targets[:, i])), label="mean target (baseline)", linestyle='dotted', color='goldenrod', alpha=.75)
        axes[i].legend()
        
    plt.suptitle(f'{args.data_type} Predictions vs Targets, loss: {np.mean(test_losses):.2f}, RMSE: {np.mean(rmse):.2f}, correlation: {np.mean(correlations):.2f} baseline RMSE: {np.mean(baseline_rmse):.2f}', fontsize=24)
    plt.tight_layout
    plt.savefig(f'reports/figures/model_results/{args.data_type}_predictions.pdf')
    if show:
        plt.show()
    plt.close()
    
    # Plot zoomed in version of each KPI
    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes = axes.flat
    start = 50
    end = 100
    
    # Get zoomed in errors
    rmse, baseline_rmse, correlation = calculate_errors(predictions[start:end], targets[start:end])
    correlations = correlation[1]
    lags = correlation[0]
    
    # Plot each KPIs
    for i in range(len(axes)):
        axes[i].title = axes[i].set_title(f'{KPI[i]}, RMSE: {rmse[i]:.2f}, correlation: {correlations[i]:.2f}, baseline RMSE: {baseline_rmse[i]:.2f}')
        axes[i].plot(np.arange(start, end, step=1), predictions[:, i][start:end], label="predicted", color='indianred', alpha=.75)
        axes[i].plot(np.arange(start, end, step=1), targets[:, i][start:end], label="target", color='royalblue', alpha=.75)
        axes[i].plot(np.arange(start, end, step=1), np.full(len(targets[start:end]), np.mean(targets[:, i][start:end])), label="mean target (baseline)", linestyle='dotted', color='goldenrod', alpha=.75)
        axes[i].legend()
        
    plt.suptitle(f'Zoomed {args.data_type} Predictions vs Targets, RMSE: {np.mean(rmse):.2f}, correlation: {np.mean(correlations):.2f} baseline RMSE: {np.mean(baseline_rmse):.2f}', fontsize=24)
    plt.tight_layout
    plt.savefig(f'reports/figures/model_results/{args.data_type}_predictions_zoomed.pdf')
    if show:
        plt.show()
    plt.close()
    

def get_args():
    parser = ArgumentParser(description='Predict Model')
    parser.add_argument('--model', type=str, default='models/best_HydraMRRegressor.pt', help='Path to the model file.')
    parser.add_argument('--data', type=str, default='data/processed/features.hdf5".csv', help='Path to the data file.')
    parser.add_argument('--feature_extractors', type=str, nargs='+', default=['MultiRocketMV_50000', 'HydraMV_8_64'], help='Feature extractors to use for prediction.')
    parser.add_argument('--name_identifier', type=str, default='', help='Name identifier for the feature extractors.')
    parser.add_argument('--data_type', type=str, default='test', help='Type of data to use for prediction.', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction.')
    parser.add_argument('--plot_during', action='store_true', help='Plot predictions during prediction.')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    model = HydraMRRegressor()
    model.load_state_dict(torch.load(args.model))
    
    # Load data
    testset = Features(data_type=args.data_type, feature_extractors=args.feature_extractors, name_identifier=args.name_identifier)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    predictions, targets, test_losses = predict(model, test_loader)
    
    plot_predictions(predictions, targets, test_losses)
    