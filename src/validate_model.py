import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import ArgumentParser, Namespace
from scipy.stats import linregress

from predict_model import calculate_errors


def plot_validation(predictions, targets, args: Namespace = None):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()
        
    # Threshold the predictions to be minimum 0
    predictions = np.maximum(predictions, 0)
    
    KPI = ['Damage Index (DI)', 'Rutting Index (RUT)', 'Patching Index (PI)', 'International Rougness Index (IRI)']
    
    # Setup plot
    fig, axes = plt.subplots(2, 4, figsize=(20,10))
    axes = axes.flat
    
    fleetyeet_predictions = np.load(f'reports/figures/our_model_results/HydraMRRegressor/test_predictions.npy')
    fleetyeet_targets = np.load(f'reports/figures/our_model_results/HydraMRRegressor/test_targets.npy')
    
    # Get errors
    rmse, baseline_rmse = calculate_errors(predictions, targets)
    fleetyeet_rmse, fleetyeet_baseline = calculate_errors(fleetyeet_predictions, fleetyeet_targets)
    
    pred_fleet_error, _ = calculate_errors(predictions, fleetyeet_predictions)
    
    correlations = []
    # Plot each KPI for the predictions
    for i in range(4):
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
        
        # Plot regression fit
        slope, intercept, r_value, p_value, std_err = linregress(targets[:, i], predictions[:, i])
        # adding the regression line to the scatter plot
        axes[i].plot(targets[:, i], slope*targets[:, i] + intercept, color='royalblue', label=f'Regression Fit ' + r"($R^2 = $" + f"{r_value**2:.2f})", zorder=2)

        # Set title
        axes[i].title = axes[i].set_title(f'{KPI[i]}\nRMSE: {rmse[i]:.2f}, baseline RMSE: {baseline_rmse[i]:.2f}, correlation: {correlation:.2f}')
        axes[i].legend()
        
    # Plot each KPI for the fleetyeet results
    for i in range(4):
        correlation = np.corrcoef(fleetyeet_targets[:, i], fleetyeet_predictions[:, i])[0, 1]
        correlations.append(correlation)
        
        # Set limits
        xlim_min = min(min(fleetyeet_targets[:, i]), min(fleetyeet_predictions[:, i])) - 0.1
        ylim_min = min(min(fleetyeet_targets[:, i]), min(fleetyeet_predictions[:, i])) - 0.1
        xlim_max = max(max(fleetyeet_targets[:, i]), max(fleetyeet_predictions[:, i])) + 0.1
        ylim_max = max(max(fleetyeet_targets[:, i]), max(fleetyeet_predictions[:, i])) + 0.1
        axes[i+4].set_xlim(xlim_min, xlim_max)
        axes[i+4].set_ylim(ylim_min, ylim_max)
        
        # Plot diagonal
        axes[i+4].plot([xlim_min, xlim_max], [ylim_min, ylim_max], color='black', linestyle='dashed', linewidth=1, alpha=.75, zorder=1)
        
        # Plot mean target
        axes[i+4].axhline(np.mean(fleetyeet_targets[:, i]), color='goldenrod', linestyle='dashed', linewidth=1, alpha=.75, label='Mean Target (baseline)', zorder=1)
        
        # Plot scatter
        axes[i+4].scatter(fleetyeet_targets[:, i], fleetyeet_predictions[:, i], color='royalblue', alpha=.75, label='Predicted vs Target', zorder=0)
        # axes[i].scatter(targets[:, i], np.full(len(targets), np.mean(targets[:, i])), color='goldenrod', alpha=.75, label='Mean Target (baseline)')
        axes[i+4].set_xlabel('FleetYeet Target', fontsize=12)
        axes[i+4].set_ylabel('FleetYeet Prediction', fontsize=12)
        
                # Plot regression fit
        slope, intercept, r_value, p_value, std_err = linregress(fleetyeet_targets[:, i], fleetyeet_predictions[:, i])
        # adding the regression line to the scatter plot
        axes[i+4].plot(fleetyeet_targets[:, i], slope*fleetyeet_targets[:, i] + intercept, color='indianred', label=f'Regression Fit ' + r"($R^2 = $" + f"{r_value**2:.2f})", zorder=2)

        # Set title
        axes[i+4].title = axes[i+4].set_title(f'{KPI[i]}\nRMSE: {fleetyeet_rmse[i]:.2f}, baseline RMSE: {fleetyeet_baseline[i]:.2f}, correlation: {correlation:.2f}')
        axes[i+4].legend()
    
    plt.suptitle(f'Predictions vs Targets for both the newly trained model (top row) and the FleetYeet results (bottom row)\nRMSE (pred, fleetyeet): {pred_fleet_error.mean():.2f}')
    plt.tight_layout()
    plt.show()
    
    
def get_args(external_parser: ArgumentParser = None):
    if external_parser is None:
        parser = ArgumentParser()
    else:
        parser = external_parser
    
    if external_parser is None:
        return parser.parse_args()    
    else:
        return parser


def main(args: Namespace):
    path_to_model = Path(args.model).stem
    if path_to_model.startswith('best_'):
        path_to_model = path_to_model[5:]
    os.makedirs(f'reports/figures/model_results/{path_to_model}', exist_ok=True)
    
    # Load the fleetyeet results
    fleetyeet_predictions = np.load(f'reports/figures/our_model_results/HydraMRRegressor/test_predictions.npy')
    fleetyeet_targets = np.load(f'reports/figures/our_model_results/HydraMRRegressor/test_targets.npy')
    
    # Load the predicted results
    predictions = np.load(f'reports/figures/model_results/{path_to_model}/{args.data_type}_predictions.npy')
    targets = np.load(f'reports/figures/model_results/{path_to_model}/{args.data_type}_targets.npy')
    
    # Calculate the errors
    rmse, baseline_rmse = calculate_errors(predictions, targets)
    fleetyeet_rmse, fleetyeet_baseline_rmse = calculate_errors(fleetyeet_predictions, fleetyeet_targets)
    
    # Calculate error between fleetyeet predictions and predictions
    fleetyeet_error, _ = calculate_errors(fleetyeet_predictions, predictions)
    
    # Plot the results
    plot_validation(predictions, targets, args=args) 
    
    
if __name__ == '__main__':
    args = get_args()



