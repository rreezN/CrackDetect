import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed/w_kpis/segments.hdf5', data_type='train', gm_cols=['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2'],
                 pm_windowsize=1, rut='straight-edge', random_state=42, data_transform=None, kpi_transform=None, kpi_names = ['DI', 'RUT', 'PI', 'IRI']):
        self.data_path = data_path
        
        # Specify transform functions
        self.data_transform = data_transform
        self.kpi_transform = kpi_transform    

        # Set variables for the data
        self.windowsize = pm_windowsize # NOTE (1 or 2)This is a +- window_size in seconds (since each segment is divided into 1 second windows)
        if self.windowsize > 2 or self.windowsize < 1:
            raise ValueError('The window size must be 1 or 2 seconds')
        self.rut = rut
        self.segments = h5py.File(self.data_path, 'r')
        self.gm_cols = gm_cols
        self.gm_cols_indices = [self.segments['0']['2']['gm'].attrs[col] for col in gm_cols]
        self.kpi_names = kpi_names
        self.kpi_names_indices = [self.segments['0']['2']['kpis'][str(self.windowsize)].attrs[col] for col in kpi_names]

        # Unfold the data to (segment, second)
        permutations = []
        for key_val in self.segments.keys():
            for sec_val in self.segments[key_val].keys():
                permutations.append((key_val, sec_val))

        # Create indices for train, test and validation
        train_indices, test_indices, _, _ = train_test_split(permutations, permutations, test_size=0.2, random_state=random_state)
        train_indices, val_indices, _, _ = train_test_split(train_indices, train_indices, test_size=0.1, random_state=random_state)
        
        # Set the indices based on the data type
        if data_type == 'train':
            self.indices = train_indices
        elif data_type == 'test':
            self.indices = test_indices
        elif data_type == 'val':
            self.indices = val_indices
        else:
            raise ValueError('data_type must be either "train", "test" or "val"')
        
        self.print_arguments()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Fetches the data sample and its corresponding labels for the given index.

        Parameters:
        idx (int): The index of the data sample to fetch.

        Returns:
        tuple: A tuple containing the transformed training data and the Key Performance Indicators (KPIs) as labels. 
               The training data is a 2D array where each row corresponds to a different GM column and each column 
               corresponds to a different second in the segment. The KPIs are a 1D array where each element corresponds 
               to a different KPI.

        This function works by first determining the segment and second numbers from the indices. It then extracts the 
        relevant GM data and KPIs from the data stored in the object. If transformations have been specified for the data 
        or KPIs, these are applied. Finally, the transformed data and KPIs are returned.
        """
        # Define variable to contain the current segment data
        segment_nr = str(self.indices[idx][0])
        second_nr = str(self.indices[idx][1])

        # Specify the data of interest
        data = self.segments[segment_nr][second_nr]

        # Extract the columns of interest in the GM data
        train = data['gm'][:,tuple(self.gm_cols_indices)]

        # Extract the KPIs based on the window size
        KPIs = data['kpis'][str(self.windowsize)][self.kpi_names_indices]

        # Transform the data
        if self.data_transform:
            train = self.data_transform(train)
        if self.kpi_transform:
            KPIs = self.kpi_transform(KPIs)
        # Return
        return train.T, KPIs # train data, labels

    def print_arguments(self):
        print(f'Arguments: \n \
                    Data Path:          {self.data_path}\n \
                    Train data Columns: {self.gm_cols} \n \
                    KPI:           \n \
                        - Names:        {self.kpi_names} \n \
                        - Window Size:  {self.windowsize} \n \
                    Selected RUT:       {self.rut} \n \
                    Transform:          \n \
                        - Data:         {None if not self.data_transform else True} \n \
                        - KPI:          {None if not self.kpi_transform else True} \n \
                    ')

if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    trainset = Platoon(data_type='train', pm_windowsize=2)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    start = time.time()
    i = 0
    durations = []
    for data, target in train_loader:
        end = time.time()
        # asses the shape of the data and target
        duration = end-start
        durations += [duration]
        print(f'Index: {i}, Time: {duration}.')
        i+= 1
        start = time.time()
    print(f'Mean duration: {np.mean(durations)}')
    # Print example statistics of the last batch
    print(f'Last data shape: {data.shape}')
    print(f'Last target shape: {target.shape}')