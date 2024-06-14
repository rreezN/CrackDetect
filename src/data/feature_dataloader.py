import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np

class Features(torch.utils.data.Dataset):
    """Dataset containing the features and KPIs for the given feature extractors and KPI window size.

    Parameters:
    ----------
        data_path (str): The path to the HDF5 file containing the features and KPIs.
        feature_extractors (list[str]): The names of the feature extractors to use.
        name_identifier (str): The identifier to append to the feature extractor names.
        data_type (str): The type of data to load (train, val, test).
        kpi_window (int): The size of the KPI window (1 or 2 seconds).
        feature_transform (callable): The transform function to apply to the features.
        fold (int): The cross-validation fold to load the data from. Default is -1 (no cross-validation).
    """
    
    def __init__(self, 
                 data_path: str = 'data/processed/features.hdf5', 
                 feature_extractors: list[str] = ['MultiRocket_50000', 'Hydra_2'], 
                 name_identifier: str = '', 
                 data_type:str = 'train', 
                 kpi_window: int = 1,
                 fold: int = -1,
                 verbose: bool = True):
        
        assert kpi_window in [1, 2], 'The kpi window size must be 1 or 2 seconds'
        assert data_type in ['train', 'val', 'test'], 'The data type must be either "train", "val" or "test"'
        
        self.data_path = data_path
        self.feature_extractors = feature_extractors
        self.name_identifier = name_identifier
        self.kpi_window_size = str(kpi_window)
        self.data_type = data_type
        self.fold = fold
        
        
        self.data = h5py.File(self.data_path, 'r')
        
        # Load the feature statistics for each feature extractor
        # Will result in a list of means and stds for each feature extractor 
        # means = [[mean1, mean2, ...], [mean1, mean2, ...]], stds = [[std1, std2, ...], [std1, std2, ...]]
        self.feature_mins = []
        self.feature_maxs = []
        self.feature_means = []
        self.feature_stds = []
        
        if self.data_type != 'test':
            for i in range(len(feature_extractors)):
                name = feature_extractors[i] + f'_{name_identifier}' if name_identifier != '' else feature_extractors[i]
                if self.fold != -1:
                    data_tree_path = self.data['train'][f'fold_{self.fold}']['statistics'][name]
                else:
                    data_tree_path = self.data['train']['statistics'][name]
                self.feature_mins.append(torch.tensor(data_tree_path['min'][()]))
                self.feature_maxs.append(torch.tensor(data_tree_path['max'][()]))
                self.feature_means.append(torch.tensor(data_tree_path['mean'][()]))
                self.feature_stds.append(torch.tensor(data_tree_path['std'][()]))
            
            # Load the KPI statistics
            if self.fold != -1:
                data_tree_path = self.data['train'][f'fold_{self.fold}']['statistics']['kpis'][str(kpi_window)]
            else:
                data_tree_path = self.data['train']['statistics']['kpis'][str(kpi_window)]
            
            self.kpi_means = data_tree_path['mean'][()]
            self.kpi_stds = data_tree_path['std'][()]
            self.kpi_mins = data_tree_path['min'][()]
            self.kpi_maxs = data_tree_path['max'][()]
            
            # Unfold the data to (segment, second)
            permutations = []
            if self.fold != -1:
                segments = self.data[data_type][f'fold_{self.fold}']['segments']
            else:
                segments = self.data[data_type]['segments']
            for key_val in segments.keys():
                for sec_val in segments[key_val].keys():
                    permutations.append((key_val, sec_val))
                    
        elif self.data_type == 'test':
            for i in range(len(feature_extractors)):
                name = feature_extractors[i] + f'_{name_identifier}' if name_identifier != '' else feature_extractors[i]
                data_tree_path = self.data['train'][f'fold_{self.fold}']['statistics'][name]
                self.feature_mins.append(torch.tensor(data_tree_path['min'][()]))
                self.feature_maxs.append(torch.tensor(data_tree_path['max'][()]))
                self.feature_means.append(torch.tensor(data_tree_path['mean'][()]))
                self.feature_stds.append(torch.tensor(data_tree_path['std'][()]))
            
            # Load the KPI statistics
            data_tree_path = self.data['train'][f'fold_{self.fold}']['statistics']['kpis'][str(kpi_window)]
            
            self.kpi_means = data_tree_path['mean'][()]
            self.kpi_stds = data_tree_path['std'][()]
            self.kpi_mins = data_tree_path['min'][()]
            self.kpi_maxs = data_tree_path['max'][()]
            
            # Unfold the data to (segment, second)
            permutations = []
            segments = self.data[data_type]['segments']
            for key_val in segments.keys():
                for sec_val in segments[key_val].keys():
                    permutations.append((key_val, sec_val))
        
        # Set the indices to the permutations
        # This will be used to fetch the data samples
        # The indices are tuples of (segment, second)
        self.indices = permutations
        
        if verbose:
            self.print_arguments()
        
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the data sample and its corresponding labels for the given index.

        Parameters:
        ----------
        idx (int): The index of the data sample to fetch.

        Returns:
        tuple (torch.tensor, torch.tensor): The data sample and its corresponding labels.
        """
        
        segment_index = str(self.indices[idx][0])
        second_index = str(self.indices[idx][1])
        
        # Load the data from the HDF5 file based on fold (and if fold is -1, no cross-validation is used)
        if self.data_type != 'test':
            if self.fold != -1:
                data = self.data[self.data_type][f'fold_{self.fold}']['segments'][segment_index][second_index]
            else:
                data = self.data[self.data_type]['segments'][segment_index][second_index]
        elif self.data_type == 'test':
            data = self.data[self.data_type]['segments'][segment_index][second_index]
            
        features = torch.tensor([])
        
        # Concatenate the features from all feature extractors (as done in the Hydra paper for HydraMultiRocket)
        for i, feature_extractor in enumerate(self.feature_extractors):
            name = feature_extractor + f'_{self.name_identifier}' if self.name_identifier != '' else feature_extractor
            feats = torch.tensor(data[name][()])

            # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
            if 'hydra' in feature_extractor.lower():
                mask = feats != 0
                feats = ((feats - self.feature_means[i])*mask)/self.feature_stds[i]
            elif 'multirocket' in feature_extractor.lower():
                feats = (feats - self.feature_means[i])/self.feature_stds[i]
            else:
                raise ValueError(f'Feature extractor "{feature_extractor}" not recognized')
            
            
            # If std = 0, set nan to 0 (no information in the feature)
            # This happens to the MultiRocket features in some cases
            # The authors of the MultiRocket paper also set the transformed features to 0 in these cases
            feats = torch.nan_to_num(feats, posinf=0, neginf=0)
            
            features = torch.cat((features, feats))
            
        targets = data['kpis'][self.kpi_window_size][()]
        
        # Standardize targets
        targets = (targets - self.kpi_means)/self.kpi_stds
        
        # Convert to torch tensors, so they can be used by a torch model
        features = features.type(torch.FloatTensor)
        targets = torch.tensor(targets).type(torch.FloatTensor)
        
        return features, targets
    
    def get_data_shape(self):
        """Returns the shape of the data and target tensors.

        Returns:
        -------
            tuple: features.shape, targets.shape
        """
        features, targets = self.__getitem__(0)
        return features.shape, targets.shape
    
    def plot_data(self):
        """Plots the histograms of the GM data before and after transformation.
        """
        # Define variable to contain the current segment data
        all_data = np.array([])
        all_data_transformed = np.array([])
        for idx in tqdm(range(len(self.indices))):
            segment_nr = str(self.indices[idx][0])
            second_nr = str(self.indices[idx][1])
            data = self.data[self.data_type]['segments'][segment_nr][second_nr]
            targets = data['kpis'][self.kpi_window_size][()]
            # Standardize targets
            if len(all_data) == 0:
                all_data = targets
                all_data_transformed = (targets - self.kpi_means)/self.kpi_stds
            else:
                all_data = np.vstack((all_data, targets))
                all_data_transformed = np.vstack((all_data_transformed, (targets - self.kpi_means)/self.kpi_stds))
        
        # Create subplots 3 histograms in 1 row
        # and transformed below
        fig, axs = plt.subplots(2, all_data.shape[1], figsize=(20, 10))
        axs = axs.flat
        for i in range(all_data.shape[1]):
            axs[i].hist(all_data[:,i], bins=100, color='blue', alpha=0.7, label='Original')
            axs[i].set_title(f'{i}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            axs[i].legend()
            
            axs[i+all_data.shape[1]].hist(all_data_transformed[:,i], bins=100, color='red', alpha=0.7, label='Transformed')
            axs[i+all_data.shape[1]].set_title(f'Transformed {i}')
            axs[i+all_data.shape[1]].set_xlabel('Value')
            axs[i+all_data.shape[1]].set_ylabel('Frequency')
            axs[i+all_data.shape[1]].legend()
        
        plt.suptitle(f'Histograms of {self.data_type} targets')
        plt.show()
    
    def print_arguments(self):
        print(f'Arguments: \n \
                    Data Path:             {self.data_path} \n \
                    Data Type:             {self.data_type} \n \
                    Data length:           {self.__len__()} \n \
                    Features selected:                      \n \
                        - Names:           {[feature_extractors for feature_extractors in self.feature_extractors]} \n \
                        - KPI Window Size: {self.kpi_window_size} \n \
                    ')   

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    train_dataset = Features(data_type='train', feature_extractors=['MultiRocketMV_50000', 'HydraMV_8_64'], name_identifier='')
    train_dataset.plot_data()
    val_dataset = Features(data_type='val', feature_extractors=['MultiRocketMV_50000', 'HydraMV_8_64'], name_identifier='')
    val_dataset.plot_data()
    test_dataset = Features(data_type='test', feature_extractors=['MultiRocketMV_50000', 'HydraMV_8_64'], name_identifier='')
    test_dataset.plot_data()
    
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    start = time.time()
    i = 0
    durations = []
    iterator = tqdm(dataloader, unit='index', position=0, leave=False)
    for features, targets in iterator:
        end = time.time()
        # assess the shape of the data and target
        duration = end-start
        durations += [duration]
        # print(f'Index: {i}, Time: {duration}.')
        i+= 1
        start = time.time()
        iterator.set_description(f'Index: {i}, Avg Time: {np.mean(duration):.3f}.')
    print(f'Mean duration: {np.mean(durations):.5f}')
    
    # Print example statistics of the last batch
    print(f'Last data shape: {features.shape}')
    print(f'Last target shape: {targets.shape}') 
    
    