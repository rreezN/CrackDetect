import os

import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import plotly.express as px
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from .feature_dataloader import Features
from ..models.hydramr import HydraMRRegressor

def predict(model: torch.nn.Module, dataloader: DataLoader):
    """Run prediction for a given model and dataloader.
    
    Parameters:
    ----------
        model: model to use for prediction
        dataloader: dataloader to use for prediction
    
    Returns:
    -------
        all_predictions (torch.Tensor), all_targets (torch.Tensor), test_losses (ndarray): predictions, targets and losses for the test set

    """
    model.eval()
    
    all_predictions = torch.tensor([])
    all_targets = torch.tensor([])
    losses = np.array([])
    
    iterator = tqdm(dataloader, unit="batch", position=0, leave=False)
    kpi_means = torch.tensor(dataloader.dataset.kpi_means)
    kpi_stds = torch.tensor(dataloader.dataset.kpi_stds)
    
    for data, targets in iterator:
        output = model(data)
        
        # Convert back from standardized to original scale
        output = ((output * kpi_stds) + kpi_means)
        targets = ((targets * kpi_stds) + kpi_means)
        
        loss_fn = nn.MSELoss()
        all_predictions = torch.cat((all_predictions, output), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)
        
        loss = loss_fn(output, targets).item()
        losses = np.append(losses, loss)
        
        iterator.set_description(f'Overall RMSE (loss): {np.sqrt(losses.mean()):.2f} Batch RMSE (loss): {np.sqrt(loss):.2f}')
    
    return all_predictions, all_targets, losses


def model_loader(feature_path, fold, feature_extractors, batch_size, model_path, hidden_dim, model_depth, batch_norm):
    datasets = {
        "train": Features(feature_path, data_type="train", feature_extractors=feature_extractors, fold=fold),
        "val": Features(feature_path, data_type="val", feature_extractors=feature_extractors, fold=fold),
        "test": Features(feature_path, data_type="test", feature_extractors=feature_extractors, fold=fold)

    }

    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=False, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    # Load the model
    input_shape, target_shape = datasets["train"].get_data_shape()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HydraMRRegressor(in_features=input_shape[0], out_features=target_shape[0], hidden_dim=hidden_dim, model_depth=model_depth, batch_norm=batch_norm)
    model.load_state_dict(torch.load(model_path, map_location=device))

    
    return datasets, dataloaders, model


def get_all_predictions(datasets, dataloaders, model):
    all_predictions = torch.tensor([])
    all_targets = torch.tensor([])

    for dataset_type in ["train", "val", "test"]:
        predictions, targets, losses = predict(model, dataloaders[dataset_type])
        all_predictions = torch.cat((all_predictions, predictions), dim=0)
        all_targets = torch.cat((all_targets, targets), dim=0)
        print(f"{dataset_type} RMSE: {np.mean(np.sqrt(losses)):.2f}")

    data_seg_sec_list = datasets["train"].indices + datasets["val"].indices + datasets["test"].indices
    data_seg_sec_list

    # Initialize the nested dictionary
    road_predictions = {}

    # Iterate through data_seg_sec_list and all_predictions to populate the dictionary
    for idx, (outer_key, inner_key) in enumerate(data_seg_sec_list):
        outer_key, inner_key = int(outer_key), int(inner_key)
        if outer_key not in road_predictions:
            road_predictions[outer_key] = {}
        road_predictions[outer_key][inner_key] = all_predictions[idx].tolist()

    return road_predictions


def get_predictions_for_POIs(weights_hh, locations_hh, road_predictions):
    lat = []
    lon = []
    preds_POI_di = []
    preds_POI_rut = []
    preds_POI_pi = []
    preds_POI_iri = []

    preds_POI_di_std = []
    preds_POI_rut_std = []
    preds_POI_pi_std = []
    preds_POI_iri_std = []

    # Convert list of POIs into sorted list
    int_list = [int(x) for x in weights_hh.keys()]
    sorted_idx = sorted(int_list)

    for POI_hh in sorted_idx: # weights_hh.keys():
        loc = locations_hh[POI_hh]
        lon.append(loc[0])
        lat.append(loc[1])

        current_preds_POI_weight = []

        current_preds_POI_di = []
        current_preds_POI_rut = []
        current_preds_POI_pi = []
        current_preds_POI_iri = []

        for MOI in weights_hh[POI_hh]:
            MOI_segment = MOI[2]
            MOI_second = MOI[3]
            current_preds_POI_weight.append(MOI[4])
            current_preds_POI_di.append(road_predictions[MOI_segment][MOI_second][0])
            current_preds_POI_rut.append(road_predictions[MOI_segment][MOI_second][1])
            current_preds_POI_pi.append(road_predictions[MOI_segment][MOI_second][2])
            current_preds_POI_iri.append(road_predictions[MOI_segment][MOI_second][3])
        
        preds_POI_di.append(sum([value * weight for value, weight in zip(current_preds_POI_di, current_preds_POI_weight)]))
        preds_POI_rut.append(sum([value * weight for value, weight in zip(current_preds_POI_rut, current_preds_POI_weight)]))
        preds_POI_pi.append(sum([value * weight for value, weight in zip(current_preds_POI_pi, current_preds_POI_weight)]))
        preds_POI_iri.append(sum([value * weight for value, weight in zip(current_preds_POI_iri, current_preds_POI_weight)]))

        preds_POI_di_std.append(np.std(current_preds_POI_di))
        preds_POI_rut_std.append(np.std(current_preds_POI_rut))
        preds_POI_pi_std.append(np.std(current_preds_POI_pi))
        preds_POI_iri_std.append(np.std(current_preds_POI_iri))
    
    return lat, lon, preds_POI_di, preds_POI_rut, preds_POI_pi, preds_POI_iri, preds_POI_di_std, preds_POI_rut_std, preds_POI_pi_std, preds_POI_iri_std


def get_kpis_for_POIs(weights_hh, locations_hh, gm):
    lat = []
    lon = []
    kpis_POI_di = []
    kpis_POI_rut = []
    kpis_POI_pi = []
    kpis_POI_iri = []

    kpis_POI_di_std = []
    kpis_POI_rut_std = []
    kpis_POI_pi_std = []
    kpis_POI_iri_std = []

    # Convert list of POIs into sorted list
    int_list = [int(x) for x in weights_hh.keys()]
    sorted_idx = sorted(int_list)

    for POI_hh in sorted_idx: # weights_hh.keys():
        loc = locations_hh[POI_hh]
        lon.append(loc[0])
        lat.append(loc[1])

        current_kpis_POI_weight = []

        current_kpis_POI_di = []
        current_kpis_POI_rut = []
        current_kpis_POI_pi = []
        current_kpis_POI_iri = []

        for MOI in weights_hh[POI_hh]:
            MOI_segment = MOI[2]
            MOI_second = MOI[3]
            current_kpis_POI_weight.append(MOI[4])
            current_kpis_POI_di.append(gm[str(MOI_segment)][str(MOI_second)]["kpis"]["2"][0])
            current_kpis_POI_rut.append(gm[str(MOI_segment)][str(MOI_second)]["kpis"]["2"][1])
            current_kpis_POI_pi.append(gm[str(MOI_segment)][str(MOI_second)]["kpis"]["2"][2])
            current_kpis_POI_iri.append(gm[str(MOI_segment)][str(MOI_second)]["kpis"]["2"][3])
        
        kpis_POI_di.append(sum([value * weight for value, weight in zip(current_kpis_POI_di, current_kpis_POI_weight)]))
        kpis_POI_rut.append(sum([value * weight for value, weight in zip(current_kpis_POI_rut, current_kpis_POI_weight)]))
        kpis_POI_pi.append(sum([value * weight for value, weight in zip(current_kpis_POI_pi, current_kpis_POI_weight)]))
        kpis_POI_iri.append(sum([value * weight for value, weight in zip(current_kpis_POI_iri, current_kpis_POI_weight)]))

        kpis_POI_di_std.append(np.std(current_kpis_POI_di))
        kpis_POI_rut_std.append(np.std(current_kpis_POI_rut))
        kpis_POI_pi_std.append(np.std(current_kpis_POI_pi))
        kpis_POI_iri_std.append(np.std(current_kpis_POI_iri))
    
    return lat, lon, kpis_POI_di, kpis_POI_rut, kpis_POI_pi, kpis_POI_iri, kpis_POI_di_std, kpis_POI_rut_std, kpis_POI_pi_std, kpis_POI_iri_std
