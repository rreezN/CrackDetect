import torch
import glob
import os
import csv
import pandas as pd

class Platoon(torch.utils.data.Dataset):
    def __init__(self, data_path='data/processed', windowsize=10, transform=None):
        self.windowsize = windowsize
        self.aran = sorted(glob.glob(data_path + '/aran/*.csv'))
        self.gm = sorted(glob.glob(data_path + '/gm/*.csv'))
        self.gopro = sorted(glob.glob(data_path + '/gopro/*.csv'))
        self.p79 = sorted(glob.glob(data_path + '/p79/*.csv'))

    def __len__(self):
        return len(self.aran)
    
    def __getitem__(self, idx):
        # Read data
        aran = pd.read_csv(self.aran[idx], sep=';', encoding='unicode_escape', engine='pyarrow')
        gm = pd.read_csv(self.gm[idx], sep=';', encoding='unicode_escape', engine='pyarrow')
        gopro = pd.read_csv(self.gopro[idx], sep=';', encoding='unicode_escape', engine='pyarrow')
        p79 = pd.read_csv(self.p79[idx], sep=';', encoding='unicode_escape', engine='pyarrow')

        # Split data into windows
        windows = len(aran) // self.windowsize
        aran_split = [aran[i*self.windowsize:(i+1)*self.windowsize] for i in range(windows)]
        gm_split = [gm[i*self.windowsize:(i+1)*self.windowsize] for i in range(windows)]
        gopro_split = [gopro[i*self.windowsize:(i+1)*self.windowsize] for i in range(windows)]
        p79_split = [p79[i*self.windowsize:(i+1)*self.windowsize] for i in range(windows)]

        # Calculate KPIs for each window
        KPIs = []
        for i in range(windows):
            KPIs.append(self.calculateKPIs(aran_split[i]))
        print(KPIs)

        
        # Return (windows, acceleration), (KPIs)
        return KPIs

    def calculateKPIs(self, df):
        # damage index
        KPI_DI = self.damageIndex(df)
        # rutting index
        KPI_RUT = self.ruttingMean(df)
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
        # LCS = df['Revner På Langs Små (m)'].fillna(0)
        # LCM = df['Revner På Langs Middelstore (m)'].fillna(0)
        # LCL = df['Revner På Langs Store (m)'].fillna(0)
        # TCS = df['Transverse Low (m)'].fillna(0)
        # TCM = df['Transverse Medium (m)'].fillna(0)
        # TCL = df['Transverse High (m)'].fillna(0)
        # return (LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1)
        return 0

    @staticmethod
    def alligatorSum(df):
        ACS = 0
        ACM = 0
        ACL = 0
        return (3*ACS + 4*ACM + 5*ACL)**(0.3)
    
    @staticmethod
    def potholeSum(df):
        PAS = 0
        PAM = 0
        PAL = 0
        PAD = 0
        return (5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1)

    @staticmethod
    def ruttingMean(df):
        RDL = 0
        RDR = 0
        return ((RDL +RDR)/2)**(0.5)

    @staticmethod
    def iriMean(df):
        IRL = 0
        IRR = 0
        return ((IRL + IRR)/2)**(0.2)

    @staticmethod   
    def patchingSum(df):
        # LCSe = df['Revner På Langs Sealed (m)'].fillna(0)
        # TCSe = df['Transverse Sealed (m)'].fillna(0)
        # return (LCSe**2 + 2*TCSe)**(0.1)
        return 0


if __name__ == '__main__':
    data = Platoon()

    for i in range(len(data)):
        print(data[i])