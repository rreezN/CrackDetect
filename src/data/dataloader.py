import torch
import glob
import os
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed', windowsize=10, rut='straight-edge', transform=None):
        self.windowsize = windowsize # TODO when data has been resampled and shiz, we need to ensure that the window size corresponds to the number of meters in the data
        self.rut = rut
        self.aran = sorted(glob.glob(data_path + '/aran/*.csv'))
        self.gm = sorted(glob.glob(data_path + '/gm/*.csv'))
        # self.gopro = sorted(glob.glob(data_path + '/gopro/*.csv'))
        # self.p79 = sorted(glob.glob(data_path + '/p79/*.csv'))

    def __len__(self):
        return len(self.aran)
    
    def __getitem__(self, idx):
        # Read data
        aran = pd.read_csv(self.aran[idx], sep=';', encoding='utf8', engine='pyarrow').fillna(0)
        gm = pd.read_csv(self.gm[idx], sep=';', encoding='utf8', engine='pyarrow')
        # gopro = pd.read_csv(self.gopro[idx], sep=';', encoding='unicode_escape', engine='pyarrow')
        # p79 = pd.read_csv(self.p79[idx], sep=';', encoding='utf8', engine='pyarrow')

        # Get row idx where difference between distance is either the exact window size or just over
        indices = self.calculateWindowIndices(aran['distance'])

        """
        TODO CHECK IT. Vi skal sørge for at KPI'erne bliver udregne korrekt

        TODO Husk at rette IRI udregningerne til at være baseret på p79 data - hvis det er det vi gerne vil
             
             Det kan evt. også være data fra cph_zp_hh(vh).csv. som så bare skal integreres i vores pipeline. Den indeholder IRI, MPD og RUT data
             med en spatial-resolution på 10m.

             Eller skal den måske regnes manuelt ud fra p79 data?
        """ 
        # Calculate KPIs for each window
        KPIs = np.array([self.calculateKPIs(aran[indices[i-1]:val], rut=self.rut) for (i, val) in list(enumerate(indices))[1:]])
        
        # Split other data correspondingly
        # p79_split = [p79[indices[i-1]:val] for (i, val) in list(enumerate(indices))[1:]]
        gm_split = [gm[indices[i-1]:val] for (i, val) in list(enumerate(indices))[1:]]

        train = [df['acc.xyz_2'].tolist() for df in gm_split] # NOTE look into more variables in the future
        return train, KPIs # train data, labels

    def calculateWindowIndices(self, df):
        indices = [0]
        count = 1
        for i, cur_distance in enumerate(df):
            if cur_distance >= self.windowsize*count: # NOTE windowsize is in meters
                indices.append(i)
                count += 1
        return indices

    def calculateKPIs(self, df, rut='straight-edge'):
        # damage index
        KPI_DI = self.damageIndex(df)
        # rutting index
        KPI_RUT = self.ruttingMean(df, rut)
        # patching index
        PI = self.patchingSum(df)
        # IRI
        IRI = self.iriMean(df)
        return [KPI_DI, KPI_RUT, PI, IRI]

    def damageIndex(self, df):
        crackingsum = self.crackingSum(df)
        alligatorsum = self.alligatorSum(df)
        potholessum = self.potholeSum(df)
        DI = crackingsum + alligatorsum + potholessum
        return DI

    @staticmethod
    def crackingSum(df):
        """
        Conventional/longitudinal and transverse cracks are reported as length. 
        """
        LCS = df['Revner På Langs Små (m)']
        LCM = df['Revner På Langs Middelstore (m)']
        LCL = df['Revner På Langs Store (m)']
        TCS = df['Transverse Low (m)']
        TCM = df['Transverse Medium (m)']
        TCL = df['Transverse High (m)']
        return np.mean((LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1))
    
    @staticmethod
    def alligatorSum(df):
        """
        alligator cracks are computed as area of the pavement affected by the damage
        """
        ACS = df['Krakeleringer Små (m²)']
        ACM = df['Krakeleringer Middelstore (m²)']
        ACL = df['Krakeleringer Store (m²)']
        return np.mean((3*ACS + 4*ACM + 5*ACL)**(0.3))
    
    @staticmethod
    def potholeSum(df):
        PAS = df['Slaghuller Max Depth Low (mm)']
        PAM = df['Slaghuller Max Depth Medium (mm)']
        PAL = df['Slaghuller Max Depth High (mm)']
        PAD = df['Slaghuller Max Depth Delamination (mm)']
        return np.mean((5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1))

    @staticmethod
    def ruttingMean(df, rut):
        if rut == 'straight-edge':
            RDL = df['LRUT Straight Edge (mm)']
            RDR = df['RRUT Straight Edge (mm)']
        elif rut == 'wire':
            RDL = df['LRUT Wire (mm)']
            RDR = df['RRUT Wire (mm)']
        return np.mean(((RDL +RDR)/2)**(0.5))

    @staticmethod
    def iriMean(df):
        IRL = df['Venstre IRI (m/km)']
        IRR = df['Højre IRI (m/km)']
        return np.mean(((IRL + IRR)/2)**(0.2))

    @staticmethod   
    def patchingSum(df):
        LCSe = df['Revner På Langs Sealed (m)']
        TCSe = df['Transverse Sealed (m)']
        return np.mean((LCSe**2 + 2*TCSe)**(0.1))


if __name__ == '__main__':
    data = Platoon()

    for i in range(len(data)):
        print(data[i])