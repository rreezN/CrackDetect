import torch
import glob
import os
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed/segments.hdf5', data_type='train', gm_cols=['acc.xyz_1', 'acc.xyz_2'],
                 pm_windowsize=1, rut='straight-edge', random_state=42, transform=None, only_iri=False):
        self.windowsize = pm_windowsize # NOTE This is a +- window_size in seconds (since each segment is divided into 1 second windows)
        self.rut = rut
        self.only_iri = only_iri
        self.segments = h5py.File(data_path, 'r')
        self.gm_cols = gm_cols

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
        TODO CHECK IT. Vi skal sørge for at KPI'erne bliver udregne korrekt

        TODO Husk at rette IRI udregningerne til at være baseret på p79 data - hvis det er det vi gerne vil
             
             Det kan evt. også være data fra cph_zp_hh(vh).csv. som så bare skal integreres i vores pipeline. Den indeholder IRI, MPD og RUT data
             med en spatial-resolution på 10m.

             Eller skal den måske regnes manuelt ud fra p79 data?
        """
        # Define variable to contain the current segment data
        data = self.segments[str(self.indices[idx])]
        # Get all the keys, corresponding to seconds in the segment (it is sorted in ascending order for good reason ;))
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
            gm_data.append(data[str(index)]['gm']['measurements'])
        # Stack the KPIs to a tensor
        KPIs = torch.stack(KPIs)
        # Extract the GM data
        train = self.extractData(gm_data, cols=self.gm_cols).view(len(keys), len(self.gm_cols), -1)
        return train, KPIs # train data, labels

    def extractData(self, df, cols):
        values = torch.tensor([])
        """
        TODO Refactor code here.
        for data in df:
            idx = [data.attrs(col) for col in cols]
            d = data[idx]
            values = torch.stack((values, torch.tensor(d[()])))
        """
        for data in df:
            for col in cols:
                values = torch.cat((values, torch.tensor(data[col][()])))
        # reshape to (n, len(cols))
        return values.view(-1, len(cols))

    def calculateKPIs(self, df, only_iri=False):
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
        cols = ['Revner På Langs Små (m)', 'Revner På Langs Middelstore (m)', 'Revner På Langs Store (m)',
                'Transverse Low (m)', 'Transverse Medium (m)', 'Transverse High (m)']
        df = self.extractData(df, cols=cols)
        LCS = df[:, 0]
        LCM = df[:, 1]
        LCL = df[:, 2]
        TCS = df[:, 3]
        TCM = df[:, 4]
        TCL = df[:, 5]
        return ((LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1)).mean()
    
    def alligatorSum(self, df):
        """
        alligator cracks are computed as area of the pavement affected by the damage
        """
        cols = ['Krakeleringer Små (m²)', 'Krakeleringer Middelstore (m²)', 'Krakeleringer Store (m²)']
        df = self.extractData(df, cols=cols)
        ACS = df[:, 0]
        ACM = df[:, 1]
        ACL = df[:, 2]
        return ((3*ACS + 4*ACM + 5*ACL)**(0.3)).mean()
    
    def potholeSum(self, df):
        cols = ['Slaghuller Max Depth Low (mm)', 'Slaghuller Max Depth Medium (mm)', 
                'Slaghuller Max Depth High (mm)', 'Slaghuller Max Depth Delamination (mm)']
        df = self.extractData(df, cols=cols)
        PAS = df[:, 0]
        PAM = df[:, 1]
        PAL = df[:, 2]
        PAD = df[:, 3]
        return ((5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1)).mean()

    def ruttingMean(self, df):
        if self.rut == 'straight-edge':
            cols = ['LRUT Straight Edge (mm)', 'RRUT Straight Edge (mm)']
            df = self.extractData(df, cols=cols)
            RDL = df[:, 0]
            RDR = df[:, 1]
        elif self.rut == 'wire':
            cols = ['LRUT Wire (mm)', 'RRUT Wire (mm)']
            df = self.extractData(df, cols=cols)
            RDL = df[:, 0]
            RDR = df[:, 1]
        return (((RDL +RDR)/2)**(0.5)).mean()

    def iriMean(self, df):
        cols = ['Venstre IRI (m_km)', 'Højre IRI (m_km)']
        df = self.extractData(df, cols=cols)
        IRL = df[:, 0]
        IRR = df[:, 1]
        return (((IRL + IRR)/2)**(0.2)).mean()
       
    def patchingSum(self, df):
        cols = ['Revner På Langs Sealed (m)', 'Transverse Sealed (m)']
        df = self.extractData(df, cols=cols)
        LCSe = df[:, 0]
        TCSe = df[:, 1]
        return ((LCSe**2 + 2*TCSe)**(0.1)).mean()


if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    trainset = Platoon(data_type='train', pm_windowsize=2)
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=0)

    start = time.time()
    i = 0
    for data_segment, target_segment in train_loader:
        end = time.time()
        print(f'Index: {i}, Time: {end-start}')
        start = time.time()
        i+= 1