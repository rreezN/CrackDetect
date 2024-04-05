import torch
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed/segments.hdf5', data_type='train', gm_cols=['acc.xyz_1', 'acc.xyz_2'],
                 pm_windowsize=1, rut='straight-edge', random_state=42, transform=None, only_iri=False):
        self.windowsize = pm_windowsize # NOTE This is a +- window_size in seconds (since each segment is divided into 1 second windows)
        self.rut = rut
        self.only_iri = only_iri
        self.segments = h5py.File(data_path, 'r')
        self.gm_cols = gm_cols
        # NOTE DO NOT CHANGE THE ORDER OF THE KEYS IN THE COLUMN_DICT
        self.column_dict = {'Revner På Langs Små (m)': 0, 'Revner På Langs Middelstore (m)': 1, 'Revner På Langs Store (m)': 2, # crackingsum
                            'Transverse Low (m)': 3, 'Transverse Medium (m)': 4, 'Transverse High (m)': 5, 
                            'Krakeleringer Små (m²)': 6, 'Krakeleringer Middelstore (m²)': 7, 'Krakeleringer Store (m²)': 8, # alligatorsum
                            'Slaghuller Max Depth Low (mm)': 9, 'Slaghuller Max Depth Medium (mm)': 10, 'Slaghuller Max Depth High (mm)': 11, 'Slaghuller Max Depth Delamination (mm)': 12,  # potholesum
                            'LRUT Straight Edge (mm)': 13, 'RRUT Straight Edge (mm)': 14, 'LRUT Wire (mm)': 15, 'RRUT Wire (mm)': 16, # ruttingmean
                            'Venstre IRI (m/km)': 17, 'Højre IRI (m/km)': 18, # irimean
                            'Revner På Langs Sealed (m)': 19, 'Transverse Sealed (m)': 20} # patchingsum
        
        # Create indices for train, test and validation
        keys = sorted([int(i) for i in list(self.segments.keys())])
        train_indices, test_indices, _, _ = train_test_split(keys, keys, test_size=0.2, random_state=random_state)
        train_indices, val_indices, _, _ = train_test_split(train_indices, train_indices, test_size=0.1, random_state=random_state)
        
        if data_type == 'train':
            self.indices = train_indices
        elif data_type == 'test':
            self.indices = test_indices
        elif data_type == 'val':
            self.indices = val_indices
        else:
            raise ValueError('data_type must be either "train", "test" or "val"')

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        TODO CHECK IT. Vi skal sørge for at KPI'erne bliver udregnet korrekt

        TODO Husk at rette IRI udregningerne til at være baseret på p79 data - hvis det er det vi gerne vil
             
             Det kan evt. også være data fra cph_zp_hh(vh).csv. som så bare skal integreres i vores pipeline. Den indeholder IRI, MPD og RUT data
             med en spatial-resolution på 10m.

             Eller skal den måske regnes manuelt ud fra p79 data?
        """
        # Define variable to contain the current segment data
        data = self.segments[str(self.indices[idx])]
        # Get all the keys, corresponding to seconds in the segment (it is sorted in ascending order for good reason ;)) NOTE we take the int as the keys are strings, and sorts them character-wise
        keys = sorted([int(x) for x in list(data.keys())])[self.windowsize:-self.windowsize] # Remove the first and last windows to avoid edge cases
        # Calculate KPIs for each second
        KPIs = []
        gm_data = []
        # Loop through each second in the segment
        for index in keys:
            # variable to contain the ARAN window from which the label will be extracted
            aran_data_ws = []
            for i in range(index-self.windowsize, index+self.windowsize+1):
                aran_data_ws.append(data[str(i)]['aran'])
            # Calculate KPIs for the current second
            KPIs += [self.calculateKPIs(aran_data_ws, only_iri=self.only_iri)]
            # Save the corresponding GM data for the current second
            gm_data.append(data[str(index)]['gm'])
        # Stack the KPIs to a tensor
        KPIs = torch.stack(KPIs).T
        # Extract the GM data
        train = self.extractData(gm_data, cols=self.gm_cols).view(len(self.gm_cols), -1)
        return train, KPIs # train data, labels

    def extractData(self, df_list, cols):
        values = []
        idx = tuple(df_list[0].attrs[col] for col in cols)
        for df in df_list:
            d = df[()][:, idx]
            values.append(torch.tensor(d))
        values = torch.vstack(values)
        return values

    def calculateKPIs(self, df_list, only_iri=False):
        df = self.extractData(df_list, cols=list(self.column_dict.keys()))
        # damage index
        KPI_DI = self.damageIndex(df)
        # rutting index
        KPI_RUT = self.ruttingMean(df)
        # patching index
        PI = self.patchingSum(df)
        # IRI
        IRI = self.iriMean(df)

        if only_iri:
            return IRI
        
        return torch.tensor((KPI_DI, KPI_RUT, PI, IRI))

    def damageIndex(self, df):
        crackingsum = self.crackingSum(df)
        alligatorsum = self.alligatorSum(df)
        potholessum = self.potholeSum(df)
        DI = crackingsum + alligatorsum + potholessum
        return DI

    def crackingSum(self, df):
        """
        Conventional/longitudinal and transverse cracks are reported as length. 
        """
        LCS = df[:, self.column_dict['Revner På Langs Små (m)']]
        LCM = df[:, self.column_dict['Revner På Langs Middelstore (m)']]
        LCL = df[:, self.column_dict['Revner På Langs Store (m)']]
        TCS = df[:, self.column_dict['Transverse Low (m)']]
        TCM = df[:, self.column_dict['Transverse Medium (m)']]
        TCL = df[:, self.column_dict['Transverse High (m)']]
        return ((LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1)).mean()
    
    def alligatorSum(self, df):
        """
        alligator cracks are computed as area of the pavement affected by the damage
        """
        ACS = df[:, self.column_dict['Krakeleringer Små (m²)']]
        ACM = df[:, self.column_dict['Krakeleringer Middelstore (m²)']]
        ACL = df[:, self.column_dict['Krakeleringer Store (m²)']]
        return ((3*ACS + 4*ACM + 5*ACL)**(0.3)).mean()
    
    def potholeSum(self, df):
        PAS = df[:, self.column_dict['Slaghuller Max Depth Low (mm)']]
        PAM = df[:, self.column_dict['Slaghuller Max Depth Medium (mm)']]
        PAL = df[:, self.column_dict['Slaghuller Max Depth High (mm)']]
        PAD = df[:, self.column_dict['Slaghuller Max Depth Delamination (mm)']]
        return ((5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1)).mean()

    def ruttingMean(self, df):
        if self.rut == 'straight-edge':
            RDL = df[:, self.column_dict['LRUT Straight Edge (mm)']]
            RDR = df[:, self.column_dict['RRUT Straight Edge (mm)']]
        elif self.rut == 'wire':
            RDL = df[:, self.column_dict['LRUT Wire (mm)']]
            RDR = df[:, self.column_dict['RRUT Wire (mm)']]
        return (((RDL +RDR)/2)**(0.5)).mean()

    def iriMean(self, df):
        IRL = df[:, self.column_dict['Venstre IRI (m/km)']]
        IRR = df[:, self.column_dict['Højre IRI (m/km)']]
        return (((IRL + IRR)/2)**(0.2)).mean()
       
    def patchingSum(self, df):
        LCSe = df[:, self.column_dict['Revner På Langs Sealed (m)']]
        TCSe = df[:, self.column_dict['Transverse Sealed (m)']]
        return ((LCSe**2 + 2*TCSe)**(0.1)).mean()

def create_batches(data, targets, batch_size):
    num_batches = len(data) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx], targets[start_idx:end_idx]

    if len(data) % batch_size != 0:
        start_idx = num_batches * batch_size
        yield data[start_idx:], targets[start_idx:]

if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader
    batch_size = 10
    trainset = Platoon(data_type='train', pm_windowsize=2)
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=0)

    start = time.time()
    i = 0
    durations = []
    for data_segment, target_segment in train_loader:
        print(data_segment.shape)
        for data, target in create_batches(data_segment, target_segment, batch_size):
            end = time.time()
            print(data.shape)
            # asses the shape of the data and target
            # assert data_segment.shape[0] == target_segment.shape[0]
            # ensure last dim in data is 250
            # assert data_segment.shape[2] == 250
            duration = end-start
            # durations += [duration]
            print(f'Index: {i}, Time: {duration}.')
            i+= 1
            # start = time.time()
    print(f'Mean duration: {np.mean(durations)}')
    # Print example statistics of the last batch
    print(f'Last data shape: {data_segment.shape}')
    print(f'Last target shape: {target_segment.shape}')