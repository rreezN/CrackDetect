import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random


# Read csv file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# 