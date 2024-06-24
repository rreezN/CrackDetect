import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import numpy.typing as npt

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from argparse import ArgumentParser, Namespace

from src.data.dataloader import Platoon
from src.util.utils import set_all_seeds
from src.models.hydra.hydra import Hydra
from src.models.multirocket.multirocket import MultiRocket
from src.models.hydra.hydra_multivariate import HydraMultivariate
from src.models.multirocket.multirocket_multivariate import MultiRocketMultivariate


# ======================================================================================================================
#               Feature extraction from all data loaders and feature extractors
# ======================================================================================================================

def extract_all_train_features(feature_extractors: list[nn.Module], 
                               data_loaders: list[DataLoader], 
                               segment_file: h5py.File,
                               args: Namespace, 
                               cross_validation_fold: int = -1,
                               calculate_statistics: list[bool] = [True, False]):
    """Extracts features from all data loaders using all feature extractors and saves them to a hdf5 file.
    
    Parameters:
    ----------
        feature_extractors (list[nn.Module]): List of feature extractors to use [multi_rocket, hydra]
        data_loaders (list[DataLoader]): List of data loaders to extract features from [train, test, val]
        segment_file (h5py.File): File to store the extracted features and targets
        cross_validation_fold (int): The cross-validation fold to extract features for. Default -1 (no cross-validation)
    """
    
    
    with h5py.File('data/processed/features.hdf5', 'a') as f:
        # go into data_loader
        # go into fold
        # go into segments
        # go into second
        # go into feature_extractor
        # save data
        for i, data_loader in enumerate(data_loaders):
            data_type = 'train' if calculate_statistics[i] else 'val'
            data_loader_subgroup = f.require_group(data_type)
            
            # Check if we are cross validating
            # If so, create a segment subgroup for the fold
            # Else, create a subgroup for the data_loader
            if cross_validation_fold != -1:
                fold_subgroup = data_loader_subgroup.require_group(f"fold_{cross_validation_fold}")
                segments_subgroup = fold_subgroup.require_group("segments")  
            else:
                segments_subgroup = data_loader_subgroup.require_group("segments")
            
            for feature_extractor in feature_extractors:
                mean, sample_variance, running_min, running_max, all_targets, s = extract_features_from_extractor(feature_extractor=feature_extractor, 
                                                                                                                  data_loader=data_loader, 
                                                                                                                  subgroup=segments_subgroup, 
                                                                                                                  segment_file=segment_file,
                                                                                                                  calculate_statistics=calculate_statistics[i],
                                                                                                                  args=args
                                                                                                                  )
                
                # Skip if we are not calculating statistics (train, test)
                if not calculate_statistics[i]:
                    continue
                
                if cross_validation_fold != -1:
                    statistics_subgroup = fold_subgroup.require_group("statistics")
                else:
                    statistics_subgroup = data_loader_subgroup.require_group("statistics")
                
                # Save feature and target statistics from training data
                name = feature_extractor.name + f"_{args.name_identifier}" if args.name_identifier != '' else feature_extractor.name
                
                # delete existing subgroup (to overwrite it with new data) if it exists
                if name in statistics_subgroup.keys():
                    del statistics_subgroup[name]
                    
                feature_extractor_subgroup = statistics_subgroup.require_group(name)
                feature_extractor_subgroup.create_dataset("used_cols", data=data_loader.dataset.gm_cols)
                feature_extractor_subgroup.create_dataset("mean", data=mean)
                feature_extractor_subgroup.create_dataset("std", data=torch.sqrt(sample_variance) + s) # add small value to avoid division by zero in hydra features https://github.com/angus924/hydra/issues/9
                feature_extractor_subgroup.create_dataset("min", data=running_min)
                feature_extractor_subgroup.create_dataset("max", data=running_max)
                
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
                

def extract_all_test_features(feature_extractors: list[nn.Module], data_loaders: list[DataLoader], segment_file: h5py.File, args: Namespace):
    """Extracts features from all data loaders using all feature extractors and saves them to a hdf5 file.
    
    Parameters:
    ----------
        feature_extractors (list[nn.Module]): List of feature extractors to use [multi_rocket, hydra]
        data_loaders (list[DataLoader]): List of data loaders to extract features from [train, test, val]
        segment_file (h5py.File): File to store the extracted features and targets
    """
    
    
    with h5py.File('data/processed/features.hdf5', 'a') as f:
        for data_loader in data_loaders:
            data_loader_subgroup = f.require_group(data_loader.dataset.data_type)
            segments_subgroup = data_loader_subgroup.require_group("segments")
            
            for feature_extractor in feature_extractors:
                mean, sample_variance, running_min, running_max, all_targets, s = extract_features_from_extractor(feature_extractor=feature_extractor, data_loader=data_loader, subgroup=segments_subgroup, segment_file=segment_file, args=args)


# ======================================================================================================================
#               Functions for computing statistics on the fly
# ======================================================================================================================

def update(existing_aggregate: tuple[int, torch.Tensor, torch.Tensor], new_value: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """ Welford's Online Algorithm, "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    For a new value new_value, compute the new count, new mean, the new M2.
    count aggregates the number of samples seen so far
    mean accumulates the mean of the entire dataset
    M2 aggregates the squared distance from the mean
    
    Parameters:
    ----------
        existing_aggregate (tuple[int, torch.Tensor, torch.Tensor]): Tuple containing the existing count, mean, and M2
        new_value (torch.Tensor): The new value to update the aggregate with
    
    Returns:
    -------
        tuple[int, torch.Tensor, torch.Tensor]
        A tuple containing the following elements:
        - count (int): The new count
        - mean (torch.Tensor): The new mean
        - M2 (torch.Tensor): The new M2
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize(existing_aggregate: tuple[int, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Welford's Online Algorithm, "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    Retrieve the mean, variance and sample variance from an aggregate

    Parameters:
    ----------
        existing_aggregate (tuple[int, torch.Tensor, torch.Tensor]): Tuple containing the existing count, mean, and M2
    
    Returns:
    -------
        tuple[int, torch.Tensor, torch.Tensor]
        A tuple containing the following elements:
        - mean (torch.Tensor): The mean of the data
        - variance (torch.Tensor): The variance of the data
        - sample_variance (torch.Tensor): The sample variance of the data
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sample_variance)

 
# ======================================================================================================================
#               Feature extraction from a single feature extractor and all data loaders
# ======================================================================================================================

def extract_features_from_extractor(feature_extractor: nn.Module, 
                                    data_loader: DataLoader, 
                                    subgroup: h5py.Group, 
                                    segment_file: h5py.File,
                                    args: Namespace,
                                    calculate_statistics: bool = False,
                                    ) -> tuple[torch.Tensor, 
                                               torch.Tensor, 
                                               torch.Tensor, 
                                               torch.Tensor, 
                                               np.ndarray, 
                                               torch.Tensor]: 
    """Extracts features from a data loader using a feature extractor and saves them to data/processed/features.hdf5.
    If the train_loader is used, the mean, std, min, and max of the features are returned to be saved in the statistics subgroup.
    Additionally, an std correction factor 's' is returned for hydra features to avoid division by zero. 
    Lastly, all targets are returned for KPI statistics.

    If calculate_statistics is false, empty tensors and ndarrays are returned for all values.

    Statistics and computed on the fly using Welford's online algorithm to avoid storing all features in memory, thus
    speeding up the feature extraction process.

    Parameters
    ----------
        feature_extractor (nn.Module): The feature extractor to use (e.g. MultiRocket, Hydra)
        data_loader (DataLoader): The data loader to extract features from (e.g. train, test, val)
        subgroup (h5py.Group): The subgroup to save the extracted features to in the hdf5 file
        segment_file (h5py.File): The file containing the segments

    Returns: 
    ---------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]
        A tuple containing the following elements:
        - mean (torch.Tensor): The mean of the extracted features. If the data loader is not the train loader, an empty tensor is returned.
        - sample_variance (torch.Tensor): The sample variance of the extracted features. If the data loader is not the train loader, an empty tensor is returned.
        - running_min (torch.Tensor): The minimum value of the extracted features. If the data loader is not the train loader, an empty tensor is returned.
        - running_max (torch.Tensor): The maximum value of the extracted features. If the data loader is not the train loader, an empty tensor is returned.
        - all_targets (np.ndarray): An array containing all accompanying targets. If the data loader is not the train loader, an empty ndarray is returned.
        - s (torch.Tensor): The std correction factor for hydra features. If the data loader is not the train loader an empty tensor is returned.  
    """
    
    all_targets = np.array([])
    
    iterator = tqdm(data_loader, unit="batch", position=0, leave=False, total=args.subset if args.subset is not None else len(data_loader))
    data_type = 'train' if calculate_statistics else 'val/test'
    iterator.set_description(f"Extracting {data_type} features from {feature_extractor.name}")
    iteration = 0
    
    running_mean = None
    running_min = None
    running_max = None
    zeros_array = None
    n = 0
    
    segments =[]
    seconds = []
    
    for data, segment_nr, second_nr in iterator:
        segments.append(segment_nr[0])
        seconds.append(second_nr[0])
        if args.subset is not None and iteration >= args.subset:
            break
        
        # Check which feature extractor is used
        # and transform the data to correct type accordingly
        if isinstance(feature_extractor, MultiRocket) or isinstance(feature_extractor, MultiRocketMultivariate):
            data = data.numpy()
        elif isinstance(feature_extractor, HydraMultivariate) or isinstance(feature_extractor, Hydra):
            data = data.type(torch.FloatTensor)
        
        # Create subgroup for segment and second
        segment_nr = segment_nr[0]
        second_nr = second_nr[0]
        
        if not segment_nr in subgroup.keys():
            subgroup.require_group(segment_nr)
        
        segment_subgroup = subgroup[segment_nr]
        
        if not second_nr in segment_subgroup.keys():
            segment_subgroup.require_group(second_nr)
            
        # Add direction, trip name and pass name as attr to segment subgroup
        segment_subgroup.attrs['direction'] = segment_file[segment_nr].attrs['direction']
        segment_subgroup.attrs['trip_name'] = segment_file[segment_nr].attrs['trip_name']
        segment_subgroup.attrs['pass_name'] = segment_file[segment_nr].attrs['pass_name']
        
        # Copy the KPIs to the second subgroup if they are not already there
        second_subgroup = segment_subgroup[second_nr]
        if not "kpis" in second_subgroup.keys():
            copy_hdf5_(data={"kpis":segment_file[segment_nr][second_nr]["kpis"]}, group=second_subgroup)
        
        # If using univariate data, squeeze the data to remove the channel dimension
        if len(args.cols) == 1 and args.all_cols == False:
            data = data.squeeze(1)
        
        # Extract features
        features = feature_extractor(data)
        features = features.flatten()
        
        if type(features) != torch.Tensor:
            features = torch.tensor(features)
            
        # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
        # Avoids going below 0
        features = torch.sqrt(torch.clamp(features, min=0))
        
        # Save features to hdf5 file and overwrite if it already exists
        name = f"{feature_extractor.name}_{args.name_identifier}" if args.name_identifier != '' else feature_extractor.name
        if name in second_subgroup.keys():
            del second_subgroup[name]
        second_subgroup.create_dataset(name, data=features)

        # We only need to compute statistics for the training data
        if calculate_statistics:
            new_value = features

            if running_mean is None:
                existing_aggregate = (0, torch.zeros_like(new_value), torch.zeros_like(new_value))
                running_min = new_value
                running_max = new_value
            else:
                existing_aggregate = (n, running_mean, running_s)

            n, running_mean, running_s = update(existing_aggregate, new_value)  # Welford's online algorithm
            running_min = torch.min(running_min, new_value)
            running_max = torch.max(running_max, new_value)

            targets1 = segment_file[segment_nr][second_nr]["kpis"]["1"][()]
            targets2 = segment_file[segment_nr][second_nr]["kpis"]["2"][()]
            targets = np.vstack((targets1, targets2))

            if all_targets.size == 0:
                all_targets = targets
            else:
                all_targets = np.vstack((all_targets, targets))

            if "hydra" in feature_extractor.name.lower():
                if zeros_array is None:
                    zeros_array = (features == 0).type(torch.FloatTensor)
                else:
                    zeros_array += (features == 0).type(torch.FloatTensor)
            
        iteration += 1

    if not calculate_statistics:
        return torch.tensor, torch.tensor, torch.tensor, torch.tensor, np.array, torch.tensor

    # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
    # Essentially, since some groups might never be selected, we add a small value to the std to avoid division by zero
    s = torch.zeros_like(running_mean)
    if "hydra" in feature_extractor.name.lower():
        denominator = args.subset if args.subset is not None else len(data_loader)
        s = (zeros_array / denominator)**4 + 1e-8
    
    # Finalize to get the mean, variance, and sample variance
    mean, _, sample_variance = finalize((n, running_mean, running_s))  # Welford's online algorithm

    return mean, sample_variance, running_min, running_max, all_targets, s

          
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

class SubsetSampler(torch.utils.data.Sampler):
    """Samples elements from the given indices sequentially, always in the same order.
    
    Parameters:
    ----------
        indices (list): A list of indices to sample from
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_args(external_parser : ArgumentParser = None):
    if external_parser is None:
        parser = ArgumentParser(description='Hydra-MR')
    else:
        parser = external_parser
    parser.add_argument('--cols', default=['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2'], nargs='+')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all_cols', action='store_true', help='Use all columns (signals) in the dataset')
    group.add_argument('--all_cols_wo_location', action='store_true', help='Use all columns except location signals')
    parser.add_argument('--feature_extractor', type=str, default='both', choices=['multirocket', 'hydra', 'both'], help='Feature extractor to use')
    parser.add_argument('--mr_num_features', type=int, default=50000, help='Number of features to extract from MultiRocket')
    parser.add_argument('--hydra_k', type=int, default=8, help='Number of kernels in each group')
    parser.add_argument('--hydra_g', type=int, default=64, help='Number of groups')
    parser.add_argument('--subset', type=int, default=None, help='Subset of data to extract features from (will not extract statistics)')
    parser.add_argument('--name_identifier', type=str, default='', help='Identifier to add to the feature extractor name')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation, Default 5')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    if external_parser is None:
        return parser.parse_args()
    else:
        return parser


def main(args: Namespace) -> None:
    set_all_seeds(args.seed)
    
    data_path = 'data/processed/w_kpis/segments.hdf5'
    
    # Assert file exists
    assert os.path.exists(data_path), f"File not found: {data_path}. Run the preprocessing script first. (src/data/make_dataset.py)"
    
    if args.all_cols:
        cols = ['gps_0', 'gps_1',                                               # Latitute and longitude
                'acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2',                          # Acceleration in x, y, z
                'spd_veh',                                                      # Vehicle speed
                'odo',                                                          # Odometer
                'acc_long', 'acc_trans', 'acc_yaw',                             # Longitudinal, transversal, and yaw acceleration
                'strg_acc', 'strg_pos',                                         # Steering acceleration and position
                'rpm', 'rpm_fl', 'rpm_fr', 'rpm_rl', 'rpm_rr',                  # RPM for each wheel
                'whl_prs_fl', 'whl_prs_fr', 'whl_prs_rl', 'whl_prs_rr',         # Wheel tire pressure for each wheel
                'whl_trq_est',                                                  # Estimated wheel torque                          
                'trac_cons',                                                    # Traction power
                'brk_trq_elec',                                                 # Brake torque
                ]
    elif args.all_cols_wo_location:
         cols = ['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2',                         # Acceleration in x, y, z
                'spd_veh',                                                      # Vehicle speed
                'acc_long', 'acc_trans', 'acc_yaw',                             # Longitudinal, transversal, and yaw acceleration
                'strg_acc', 'strg_pos',                                         # Steering acceleration and position
                'rpm', 'rpm_fl', 'rpm_fr', 'rpm_rl', 'rpm_rr',                  # RPM for each wheel
                'whl_prs_fl', 'whl_prs_fr', 'whl_prs_rl', 'whl_prs_rr',         # Wheel tire pressure for each wheel
                'whl_trq_est',                                                  # Estimated wheel torque                          
                'trac_cons',                                                    # Traction power
                'brk_trq_elec',                                                 # Brake torque
                ]
    else:
        cols = args.cols
    
    # To properly save data
    args.cols = cols
    
    # Load data
    print(f"Loading train data from {data_path}")
    train_data = Platoon(data_path=data_path, data_type='train', feature_extraction=True, gm_cols=cols)
    
    input_shape, _, _ = train_data.get_data_shape()
    
    # Create feature extractors
    if len(cols) == 1:
        multi_rocket_transformer = MultiRocket(args.mr_num_features)
        hydra_transformer = Hydra(input_shape[1], args.hydra_k, args.hydra_g)
    else:
        multi_rocket_transformer = MultiRocketMultivariate(args.mr_num_features)
        hydra_transformer = HydraMultivariate(input_shape[1], len(cols), args.hydra_k, args.hydra_g)
    
    if args.feature_extractor.lower() == 'multirocket':
        feature_extactors = [multi_rocket_transformer]
        print(f"Extracting features using {multi_rocket_transformer.name}")
    elif args.feature_extractor.lower() == 'hydra':
        feature_extactors = [hydra_transformer]
        print(f"Extracting features using {hydra_transformer.name}")
    elif args.feature_extractor.lower() == 'both':
        feature_extactors = [multi_rocket_transformer, hydra_transformer]
        print(f"Extracting features using {multi_rocket_transformer.name} and {hydra_transformer.name}")
    else:
        raise ValueError(f"Invalid feature extractor: {args.feature_extractor}, must be one of ['multirocket', 'hydra', 'both']")
        
    print(f"Extracting features from {len(cols)} columns: {cols}")
    print(f"Extracting features for {args.folds} folds")
    
    # Split into K folds
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        # Create dataloaders
        # NOTE: Must use batch_size=1, to avoid errors when extracting features
        train_loader = DataLoader(train_data, 
                                  batch_size=1, 
                                  sampler=SubsetSampler(train_idx),
                                  )
        val_loader = DataLoader(train_data, 
                                batch_size=1, 
                                sampler=SubsetSampler(val_idx),
                                )
        
        # Extract features from train and val data for each fold
        with h5py.File(data_path, 'r') as f:
            print(f'Extracting features for fold {fold+1}/{args.folds}')
            extract_all_train_features(feature_extractors=feature_extactors, data_loaders=[train_loader, val_loader], segment_file=f, args=args, cross_validation_fold=fold, calculate_statistics=[True, False])  # [multi_rocket_transformer, hydra_transformer]
    
    # Extract features from test data
    print("Loading test data")
    test_data = Platoon(data_path=data_path, data_type='test', feature_extraction=True, gm_cols=cols)
    test_loader = DataLoader(test_data, batch_size=1)
    print("Extracting features from test data")
    with h5py.File(data_path, 'r') as f:
        extract_all_test_features(feature_extractors=feature_extactors, data_loaders=[test_loader], args=args, segment_file=f)  # [multi_rocket_transformer, hydra_transformer]


if __name__ == '__main__':
    start = time()
    
    args = get_args()
    main(args)
    
    # Report time
    extraction_time = time() - start
    minutes = int(extraction_time / 60)
    seconds = int(extraction_time % 60)
    print(f"Feature extraction took {minutes} minutes and {seconds} seconds")