import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import hdf5

class Converters:
    @staticmethod
    def median_lambda(df: pd.DataFrame, column_name) -> pd.DataFrame:
        return df.groupby(df.index // 10)[column_name].median()

    # Create formula for the generic conversion formula, and add b*, b and r*, r as inputs


# Read csv file and return the data
def read_csv(file_path) -> list[str]:  
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def load_data() -> list:
    
    platoon_hh = hdf5.load('../../data/raw/ref_data/cph1_aran_hh.csv')
    platoon_vh = hdf5.load('../../data/raw/ref_data/cph1_aran_vh.csv')
    return [platoon_hh, platoon_vh]

def main():

    data = load_data()
    convert = Converters()

if __name__ == '__main__':
    # Get the data and process it
    pass