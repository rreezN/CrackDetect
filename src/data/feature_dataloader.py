import torch
import h5py
import numpy as np

class Features(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed/features.hdf5', feature_extractors=['MultiRocket_50000', 'Hydra_2'], 
                 data_type='train', kpi_window=1, feature_transform=None, kpi_transform=None):
        
        assert kpi_window in [1, 2], 'The kpi window size must be 1 or 2 seconds'
        
        self.data_path = data_path
        self.feature_extractors = feature_extractors
        self.kpi_window_size = str(kpi_window)
        
        # Specify transform functions
        self.feature_transform = feature_transform
        self.kpi_transform = kpi_transform
        
        self.data_type = data_type
        
        self.data = h5py.File(self.data_path, 'r')
        
        self.feature_means = []
        self.feature_stds = []
        for i in range(len(feature_extractors)):
            self.feature_means.append(torch.tensor(self.data['train']['statistics'][feature_extractors[i]]['mean'][()]))
            self.feature_stds.append(torch.tensor(self.data['train']['statistics'][feature_extractors[i]]['std'][()]))
        
        self.kpi_means = self.data['train']['statistics']['kpis'][str(kpi_window)]['mean'][()]
        self.kpi_stds = self.data['train']['statistics']['kpis'][str(kpi_window)]['std'][()]
        self.kpi_mins = self.data['train']['statistics']['kpis'][str(kpi_window)]['min'][()]
        self.kpi_maxs = self.data['train']['statistics']['kpis'][str(kpi_window)]['max'][()]
        
        # Unfold the data to (segment, second)
        permutations = []
        segments = self.data[data_type]['segments']
        for key_val in segments.keys():
            for sec_val in segments[key_val].keys():
                permutations.append((key_val, sec_val))
        
        self.indices = permutations
        
        self.print_arguments()
        
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the data sample and its corresponding labels for the given index.

        Parameters:
        idx (int): The index of the data sample to fetch.

        Returns:
        tuple: The data sample and its corresponding labels.
        """
        
        segment_index = str(self.indices[idx][0])
        second_index = str(self.indices[idx][1])
        
        data = self.data[self.data_type]['segments'][segment_index][second_index]
        features = torch.tensor([])
        self.preivous_bad_feats = torch.tensor([])
        for i, feature_extractor in enumerate(self.feature_extractors):
            feats = torch.tensor(data[feature_extractor][()])

            # See https://github.com/angus924/hydra/issues/9 for why this transform is necessary
            # if 'hydra' in feature_extractor.lower():
            #     mask = feats != 0
            #     feats = ((feats - self.feature_means[i])*mask)/self.feature_stds[i]
            # elif 'multirocket' in feature_extractor.lower():
            #     feats = (feats - self.feature_means[i])/self.feature_stds[i]
            # else:
            #     raise ValueError(f'Feature extractor "{feature_extractor}" not recognized')
            
            # TODO: Change this back when new features are calculated (see above)
            feats = (feats - torch.mean(self.feature_means[i]))/torch.mean(self.feature_stds[i])
            
            # If std = 0, set nan to 0 (no information in the feature)
            # feats = torch.nan_to_num(feats)
            features = torch.cat((features, feats))
            
        targets = data['kpis'][self.kpi_window_size][()]
        
        # Transform targets to be in the range [0, 1]
        # Maybe we should use normal standardization instead of this
        targets = (targets - self.kpi_means)/self.kpi_stds
        
        # This is a bit ugly
        features = features.type(torch.FloatTensor)
        targets = torch.tensor(targets).type(torch.FloatTensor)

        # TODO: Implement actual torch transforms
        # Or maybe just delete these as they are unused
        if self.feature_transform:
            features = self.feature_transform(features)
        if self.kpi_transform:
            targets = self.kpi_transform(targets)
        
        return features, targets
    
    def print_arguments(self):
        print(f'Arguments: \n \
                    Data Path:             {self.data_path} \n \
                    Data Type:             {self.data_type} \n \
                    Data length:           {self.__len__()} \n \
                    Features selected:                      \n \
                        - Names:           {[feature_extractors for feature_extractors in self.feature_extractors]} \n \
                        - KPI Window Size: {self.kpi_window_size} \n \
                    Transform:          \n \
                        - Features:        {None if not self.feature_transform else True} \n \
                        - KPI:             {None if not self.kpi_transform else True} \n \
                    ')   

if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    dataset = Features(data_type='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
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
    
    