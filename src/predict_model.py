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


def calculate_errors(predictions: torch.Tensor, targets: torch.Tensor, lags: int = 10):
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
    
    # Calculate RMSE between targets and predictions for each KPI
    RMSE = np.sqrt(np.mean(np.abs(targets - predictions)**2, axis=0))
    
    # Calculate baseline RMSE between targets and predictions for each KPI
    baseline_RMSE = np.sqrt(np.mean(np.abs(targets - np.mean(targets, axis=0))**2, axis=0))
    
    # Calculate correlation between targets and predictions for each KPI with different lags
    correlations = []
    for lag in range(1, lags):
        correlation = np.corrcoef(targets[:-lag], predictions[lag:])[0, 1]
        correlations.append(correlation)
    
    # Save best correlation
    best_correlation = np.max(correlations)
    best_correlation_lag = np.argmax(correlations) + 1
    
    return RMSE, baseline_RMSE, (best_correlation_lag, best_correlation)


def plot_predictions(predictions, targets, test_losses, show=False):
    """Plot the predictions against the targets. Also reports the RMSE and correlation between the predictions and the targets.

    Args:
        predictions (_type_): _description_
        targets (_type_): _description_
        test_losses (_type_): _description_
        show (bool, optional): _description_. Defaults to False.
    """
    
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
    
    # Plot each KPIs
    for i in range(len(axes)):
        axes[i].title = axes[i].set_title(f'{KPI[i]}, RMSE: {rmse:.2f}, correlation: {correlation[1]:.2f}, baseline RMSE: {baseline_rmse:.2f}')
        axes[i].plot(predictions[:, i], label="predicted", color='indianred', alpha=.75)
        axes[i].plot(targets[:, i], label="target", color='royalblue', alpha=.75)
        axes[i].plot(np.mean(targets, axis=1), label="mean target (baseline)", color='goldenrod', alpha=.75)
        axes[i].legend()
        
    plt.suptitle(f'Predictions vs Targets, loss: {np.mean(test_losses):.2f}, RMSE: {rmse:.2f}, correlation: {np.mean(correlation[1]):.2f} baseline RMS: {np.mean(baseline_rmse):.2f}', fontsize=24)
    plt.tight_layout
    plt.savefig('reports/figures/model_results/predictions.pdf')
    if show:
        plt.show()
    plt.close()
    
    # Plot zoomed in version of each KPI
    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes = axes.flat
    start = 50
    end = 100
    
    # TODO: Add this and make sure it works as above
    return
    # Get zoomed in errors
    rmse, baseline_rmse, correlation = calculate_errors(predictions[start:end], targets[start:end])
    
    for i in range(len(axes)):
        # TODO: Correct correlation calculation
        # calculate correlation between predictions and targets
        correlation = np.corrcoef(targets[start:end, i], predictions[start:end, i])[0, 1]
        
        # calculate RMSE between predictions and targets
        rmse= np.sqrt(np.mean(np.abs(targets[start:end, i] - predictions[start:end, i])**2))
        
        rmses.append(rmse)
        correlations.append(correlation)
        
        axes[i].title = axes[i].set_title(f'{KPI[i]}, RMSE: {rmse:.2f}, correlation: {correlation:.2f}')
        axes[i].plot(predictions[:, i], label="predicted", color='indianred', alpha=.75)
        axes[i].plot(targets[:, i], label="target", color='royalblue', alpha=.75)
        axes[i].legend()
        axes[i].set_xlim(start,end)
    
    plt.suptitle(f'Zoomed Predictions vs Targets, RMSE: {np.mean(rmses):.2f}, correlation: {np.mean(correlations):.2f}', fontsize=24)
    plt.tight_layout
    plt.savefig('reports/figures/model_results/predictions_zoomed.pdf')
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
    