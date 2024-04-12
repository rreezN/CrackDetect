from tqdm import tqdm
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from models.hydra.hydra_multivariate import HydraMultivariate
from models.multirocket.multirocket import MultiRocket
from data.dataloader import Platoon


def extract_all_features(feature_extractors:list, data_loaders:list, segment_file:h5py.File):
    with h5py.File('data/processed/features.hdf5', 'w') as f:
        for data_loader in data_loaders:
            data_loader_subgroup = f.create_group(data_loader.dataset.data_type)
            segments_subgroup = data_loader_subgroup.create_group("segments")
            statistics_subgroup = data_loader_subgroup.create_group("statistics")
            for feature_extractor in feature_extractors:
                all_features, all_targets = extract_features_from_transformer(feature_extractor, data_loader, segments_subgroup, segment_file)
                if len(all_features) == 0:
                    continue
                # Save feature and target statistics for training only
                feature_extractor_subgroup = statistics_subgroup.create_group(feature_extractor.name)
                feature_extractor_subgroup.create_dataset("mean", data=torch.mean(all_features, dim=0))
                feature_extractor_subgroup.create_dataset("std", data=torch.std(all_features, dim=0))
                min, _ = torch.min(all_features, dim=0)
                max, _ = torch.max(all_features, dim=0)
                feature_extractor_subgroup.create_dataset("min", data=min)
                feature_extractor_subgroup.create_dataset("max", data=max)
                
                if not "kpis" in statistics_subgroup.keys():
                    kpi_stat_subgroup = statistics_subgroup.create_group("kpis")
                    target_1_subgroup = kpi_stat_subgroup.create_group("1")
                    target_1_subgroup.create_dataset("mean", data=torch.mean(torch.tensor(all_targets[::2, :]), dim=0))
                    target_1_subgroup.create_dataset("std", data=torch.std(torch.tensor(all_targets[::2, :]), dim=0))
                    min, _ = torch.min(torch.tensor(all_targets[::2, :]), dim=0)
                    max, _ = torch.max(torch.tensor(all_targets[::2, :]), dim=0)
                    target_1_subgroup.create_dataset("min", data=min)
                    target_1_subgroup.create_dataset("max", data=max)
                    
                    target_2_subgroup = kpi_stat_subgroup.create_group("2")
                    target_2_subgroup.create_dataset("mean", data=torch.mean(torch.tensor(all_targets[1::2, :]), dim=0))
                    target_2_subgroup.create_dataset("std", data=torch.std(torch.tensor(all_targets[1::2, :]), dim=0))
                    min, _ = torch.min(torch.tensor(all_targets[1::2, :]), dim=0)
                    max, _ = torch.max(torch.tensor(all_targets[1::2, :]), dim=0)
                    target_2_subgroup.create_dataset("min", data=min)
                    target_2_subgroup.create_dataset("max", data=max)
                
                    

def extract_features_from_transformer(feature_extractor, data_loader, subgroup, segment_file):
    all_features = torch.tensor([])
    all_targets = np.array([])
    iterator = tqdm(data_loader, unit="batch", position=0, leave=False)
    iterator.set_description(f"Extracting {data_loader.dataset.data_type} features from {feature_extractor.name}")
    iteration = 0
    for data, segment_nr, second_nr in iterator:
        if args.subset is not None and iteration >= args.subset:
            break
        if isinstance(feature_extractor, MultiRocket):
            data = data.numpy()
        elif isinstance(feature_extractor, HydraMultivariate):
            data = data.type(torch.FloatTensor)
        segment_nr = segment_nr[0]
        second_nr = second_nr[0]
        if not segment_nr in subgroup.keys():
            subgroup.create_group(segment_nr)
        
        segment_subgroup = subgroup[segment_nr]
        
        # Add direction, trip name and pass name as attr to segment subgroup
        segment_subgroup.attrs['direction'] = segment_file[segment_nr].attrs['direction']
        segment_subgroup.attrs['trip_name'] = segment_file[segment_nr].attrs['trip_name']
        segment_subgroup.attrs['pass_name'] = segment_file[segment_nr].attrs['pass_name']
        
        if not second_nr in segment_subgroup.keys():
            segment_subgroup.create_group(second_nr)
        
        second_subgroup = segment_subgroup[second_nr]
        if not "kpis" in second_subgroup.keys():
            copy_hdf5_(data={"kpis":segment_file[segment_nr][second_nr]["kpis"]}, group=second_subgroup)
        
        features = feature_extractor(data)
        features = features.flatten()
        if type(features) != torch.Tensor:
            features = torch.tensor(features)
        second_subgroup.create_dataset(feature_extractor.name, data=features)
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
          
        
def copy_hdf5_(data, group):
    for key, value in data.items():
        if isinstance(value, h5py.Group):
            subgroup = group.create_group(key)
            # Save attributes
            for k, v in value.attrs.items():
                subgroup.attrs[k] = v
            copy_hdf5_(value, subgroup)
        else:
            dataset = group.create_dataset(key, data=value[()])
            for k, v in value.attrs.items():
                dataset.attrs[k] = v


def get_args():
    parser = ArgumentParser(description='Hydra-MR')
    parser.add_argument('--mr_num_features', type=int, default=50000)
    parser.add_argument('--hydra_input_length', type=int, default=250)
    parser.add_argument('--hydra_num_channels', type=int, default=2)
    parser.add_argument('--subset', type=int, default=None)
    # parser.add_argument('--batch_size', type=int, default=32)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    data_path = 'data/processed/w_kpis/segments.hdf5'
    
    # Load data
    train_data = Platoon(data_path=data_path, data_type='train', feature_extraction=True)
    val_data = Platoon(data_path=data_path, data_type='val', feature_extraction=True)
    test_data = Platoon(data_path=data_path, data_type='test', feature_extraction=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    # Create feature extractors
    multi_rocket_transformer = MultiRocket(args.mr_num_features)
    hydra_transformer = HydraMultivariate(args.hydra_input_length, args.hydra_num_channels)
    
    with h5py.File(data_path, 'r') as f:
        extract_all_features([multi_rocket_transformer, hydra_transformer], [train_loader, val_loader, test_loader], f)
    
    