import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from models.hydra.hydra import Hydra
from models.hydra.hydra_multivariate import HydraMultivariate
from src.models.multirocket.multirocket_multivariate import MultiRocketMultivariate
from src.models.multirocket.multirocket import MultiRocket
from data.dataloader import Platoon


def extract_all_features(feature_extractors: list[nn.Module], data_loaders: list[DataLoader], segment_file: h5py.File):
    """Extracts features from all data loaders using all feature extractors and saves them to a hdf5 file.
    
    Parameters:
    ----------
        feature_extractors (list): List of feature extractors to use [multi_rocket, hydra]
        data_loaders (list): List of data loaders to extract features from [train, test, val]
        segment_file (h5py.File): File to store the extracted features and targets
    """
    
    
    with h5py.File('data/processed/features.hdf5', 'a') as f:
        for data_loader in data_loaders:
            data_loader_subgroup = f.require_group(data_loader.dataset.data_type)
            segments_subgroup = data_loader_subgroup.require_group("segments")
            statistics_subgroup = data_loader_subgroup.require_group("statistics")
            
            for feature_extractor in feature_extractors:
                all_features, all_targets = extract_features_from_extractor(feature_extractor, data_loader, segments_subgroup, segment_file)
                
                if len(all_features) == 0:
                    continue
                
                # Save feature and target statistics from training data
                # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
                # Essentially, since some groups might never be selected, we add a small value to the std to avoid division by zero
                if "hydra" in feature_extractor.name.lower():
                    s = torch.mean((all_features==0).type(torch.FloatTensor), dim=0)**4 + 1e-8
                else:
                    s = 0
                    
                name = feature_extractor.name + f"_{args.name_identifier}" if args.name_identifier is not None else feature_extractor.name
                
                # delete existing subgroup (to overwrite it with new data) if it exists
                if name in statistics_subgroup.keys():
                    del statistics_subgroup[name]
                    
                feature_extractor_subgroup = statistics_subgroup.require_group(name)
                feature_extractor_subgroup.create_dataset("used_cols", data=data_loader.dataset.gm_cols)
                feature_extractor_subgroup.create_dataset("mean", data=torch.mean(all_features, dim=0))
                feature_extractor_subgroup.create_dataset("std", data=torch.std(all_features, dim=0)+s) # add small value to avoid division by zero in hydra features
                min, _ = torch.min(all_features, dim=0)
                max, _ = torch.max(all_features, dim=0)
                feature_extractor_subgroup.create_dataset("min", data=min)
                feature_extractor_subgroup.create_dataset("max", data=max)
                
                # Save KPI statistics
                # NOTE: To get proper KPI statistics, feature extraction must be run on the entire dataset (NOT a subset)
                if not "kpis" in statistics_subgroup.keys() and args.subset is None:
                    kpi_stat_subgroup = statistics_subgroup.require_group("kpis")
                    target_1_subgroup = kpi_stat_subgroup.require_group("1")
                    target_1_subgroup.create_dataset("mean", data=torch.mean(torch.tensor(all_targets[::2, :]), dim=0))
                    target_1_subgroup.create_dataset("std", data=torch.std(torch.tensor(all_targets[::2, :]), dim=0))
                    min, _ = torch.min(torch.tensor(all_targets[::2, :]), dim=0)
                    max, _ = torch.max(torch.tensor(all_targets[::2, :]), dim=0)
                    target_1_subgroup.create_dataset("min", data=min)
                    target_1_subgroup.create_dataset("max", data=max)
                    
                    target_2_subgroup = kpi_stat_subgroup.require_group("2")
                    target_2_subgroup.create_dataset("mean", data=torch.mean(torch.tensor(all_targets[1::2, :]), dim=0))
                    target_2_subgroup.create_dataset("std", data=torch.std(torch.tensor(all_targets[1::2, :]), dim=0))
                    min, _ = torch.min(torch.tensor(all_targets[1::2, :]), dim=0)
                    max, _ = torch.max(torch.tensor(all_targets[1::2, :]), dim=0)
                    target_2_subgroup.create_dataset("min", data=min)
                    target_2_subgroup.create_dataset("max", data=max)
                
 

def extract_features_from_extractor(feature_extractor: nn.Module, data_loader: DataLoader, subgroup: h5py.Group, segment_file: h5py.File):
    """Extracts features from a data loader using a feature extractor, saves them to a hdf5 file and returns them.

    Parameters
    ----------
        feature_extractor (nn.Module): The feature extractor to use (e.g. MultiRocket, Hydra)
        data_loader (DataLoader): The data loader to extract features from (e.g. train, test, val)
        subgroup (h5py.Group): The subgroup to save the extracted features to in the hdf5 file
        segment_file (h5py.File): The file containing the segments

    Returns:
    ---------
        torch.tensor, np.array: A tensor containing all extracted features and an array containing all accompanying targets
    """
    
    all_features = torch.tensor([])
    all_targets = np.array([])
    
    iterator = tqdm(data_loader, unit="batch", position=0, leave=False, total=args.subset if args.subset is not None else len(data_loader))
    iterator.set_description(f"Extracting {data_loader.dataset.data_type} features from {feature_extractor.name}")
    iteration = 0
    
    for data, segment_nr, second_nr in iterator:
        if args.subset is not None and iteration >= args.subset:
            break
        
        # Check which feature extractor is used
        # and transform the data to correct type accordingly
        if isinstance(feature_extractor, MultiRocket) or isinstance(feature_extractor, MultiRocketMultivariate):
            data = data.numpy()
        elif isinstance(feature_extractor, HydraMultivariate) or isinstance(feature_extractor, Hydra):
            data = data.type(torch.FloatTensor)
            
        segment_nr = segment_nr[0]
        second_nr = second_nr[0]
        
        if not segment_nr in subgroup.keys():
            subgroup.require_group(segment_nr)
        
        segment_subgroup = subgroup[segment_nr]
        
        # Add direction, trip name and pass name as attr to segment subgroup
        segment_subgroup.attrs['direction'] = segment_file[segment_nr].attrs['direction']
        segment_subgroup.attrs['trip_name'] = segment_file[segment_nr].attrs['trip_name']
        segment_subgroup.attrs['pass_name'] = segment_file[segment_nr].attrs['pass_name']
        
        if not second_nr in segment_subgroup.keys():
            segment_subgroup.require_group(second_nr)
        
        # Copy the KPIs to the second subgroup if they are not already there
        second_subgroup = segment_subgroup[second_nr]
        if not "kpis" in second_subgroup.keys():
            copy_hdf5_(data={"kpis":segment_file[segment_nr][second_nr]["kpis"]}, group=second_subgroup)
        
        # If using univariate data, squeeze the data to remove the channel dimension
        if len(args.cols) == 1:
            data = data.squeeze(1)
        
        # Extract features
        features = feature_extractor(data)
        features = features.flatten()
        
        if type(features) != torch.Tensor:
            features = torch.tensor(features)
            
        # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
        # Avoids going below 0
        features = torch.sqrt(torch.clamp(features, min=0))
        
        if f'{feature_extractor.name}_{args.name_identifier}' in second_subgroup.keys():
            del second_subgroup[f'{feature_extractor.name}_{args.name_identifier}']
            
        second_subgroup.create_dataset(f'{feature_extractor.name}_{args.name_identifier}', data=features)

        # TODO: Optimize this and avoid storing all features in memory
        if data_loader.dataset.data_type == 'train':
            if len(all_features) == 0:
                all_features = features
            else:
                all_features = torch.vstack((all_features, features))
            targets1 = segment_file[segment_nr][second_nr]["kpis"]["1"][()]
            targets2 = segment_file[segment_nr][second_nr]["kpis"]["2"][()]
            targets = np.vstack((targets1, targets2))
            if all_targets.size == 0:
                all_targets = targets
            else:
                all_targets = np.vstack((all_targets, targets))
        
        iteration += 1
            
        
    return all_features, all_targets
          
        
def copy_hdf5_(data: dict[str, h5py.Group], group: h5py.Group):
    """Copies a h5py group to another group.

    Parameters:
    ----------
        data (dict[str, h5py.Group]): The data to copy
        group (h5py.Group): The group to copy the data to
    """
    for key, value in data.items():
        if isinstance(value, h5py.Group):
            subgroup = group.require_group(key)
            # Save attributes
            for k, v in value.attrs.items():
                subgroup.attrs[k] = v
            copy_hdf5_(value, subgroup)
        else:
            dataset = group.create_dataset(key, data=value[()])
            for k, v in value.attrs.items():
                dataset.attrs[k] = v


def get_args():
    """Parses the arguments from the command line.

    Returns:
    -------
        Namespace: Arguments from the command line
    """
    parser = ArgumentParser(description='Hydra-MR')
    parser.add_argument('--cols', default=['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2'], nargs='+')
    parser.add_argument('--mr_num_features', type=int, default=50000)
    parser.add_argument('--hydra_input_length', type=int, default=250) # our input length is 250
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--name_identifier', type=str)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    data_path = 'data/processed/w_kpis/segments.hdf5'
    
    # Load data
    train_data = Platoon(data_path=data_path, data_type='train', feature_extraction=True, gm_cols=args.cols)
    val_data = Platoon(data_path=data_path, data_type='val', feature_extraction=True, gm_cols=args.cols)
    test_data = Platoon(data_path=data_path, data_type='test', feature_extraction=True, gm_cols=args.cols)
    
    # Create dataloaders
    # NOTE: Must use batch_size=1, to avoid errors when extracting features
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    # Create feature extractors
    if len(args.cols) == 1:
        multi_rocket_transformer = MultiRocket(args.mr_num_features)
        hydra_transformer = Hydra(args.hydra_input_length)
    else:
        multi_rocket_transformer = MultiRocketMultivariate(args.mr_num_features)
        hydra_transformer = HydraMultivariate(args.hydra_input_length, len(args.cols))
    
    print(f"Extracting features from {multi_rocket_transformer.name} and {hydra_transformer.name}")
    
    with h5py.File(data_path, 'r') as f:
        extract_all_features([multi_rocket_transformer, hydra_transformer], [train_loader, val_loader, test_loader], f)
    
    