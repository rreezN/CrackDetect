#%% Import libraries and define functions
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
%matplotlib widget

# Read csv file and return the data
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

#%% Load data
# Read reference data cph1_aran

# VEHICLE: ARAN
'NOTE Road Conditional data'
data_aran_hh = pd.read_csv('../../data/raw/ref_data/cph1_aran_hh.csv', sep=';', encoding='unicode_escape')
data_aran_vh = pd.read_csv('../../data/raw/ref_data/cph1_aran_vh.csv', sep=';', encoding='unicode_escape')

# VEHICLE: VIAFRIK
'NOTE Friction data'
data_fric_hh = pd.read_csv('../../data/raw/ref_data/cph1_fric_hh.csv', sep=';', encoding='unicode_escape')
data_fric_vh = pd.read_csv('../../data/raw/ref_data/cph1_fric_vh.csv', sep=';', encoding='unicode_escape')

# VEHICLE: P79
'NOTE Road Elevation Data'
data_zp_hh = pd.read_csv('../../data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape')
data_zp_vh = pd.read_csv('../../data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')
'NOTE International Roughness Index (IRI), Mean Profile Depth (MPD) and wheelpath rut depth'
data_iri_hh = pd.read_csv('../../data/raw/ref_data/cph1_iri_mpd_rut_hh.csv', sep=';', encoding='unicode_escape')
data_iri_vh = pd.read_csv('../../data/raw/ref_data/cph1_iri_mpd_rut_vh.csv', sep=';', encoding='unicode_escape')

# %% Investigate data
data_zp_hh_columns = data_zp_hh.columns
data_zp_vh_columns = data_zp_vh.columns

i,j = 1, 25
print(data_zp_hh_columns[i], data_zp_hh_columns[j])
laser_data = data_zp_hh[[data_zp_hh_columns[k] for k in range(i,j+1)]]

#%% Plot 2d plot
laser_data_lim = laser_data.iloc[0:100, :]
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot(111)
x = np.arange(0, laser_data_lim.shape[1], 1)
# Get first row of data as y
y = laser_data_lim.iloc[0, :]
# Plot data 
ax.plot(x, y, label='zs=0, zdir=z', marker='o')
ax.grid(alpha=0.5)
ax.set_ylim(30, 90)
plt.show()

# Plot all data for laser 1
plt.plot(data_zp_hh[data_zp_hh_columns[1]])
plt.show()
# %%
# Plot in 3d, x-axis as index, y-axis as column number and z-axis as distance
laser_data_ = laser_data.iloc[100:5000, :]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, laser_data_.shape[1], 1)
y = data_zp_hh[data_zp_hh_columns[0]][100:5000]
x, y = np.meshgrid(x, y)
# mm to m
z = laser_data_/1000
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlim3d(-10, 35)
ax.set_zlim3d(0, 1)
plt.show()


# %% Plot alignment parameters 
fig = plt.figure()
x = data_aran_hh['Lon']
y = data_aran_hh['Lat']
plt.plot(x, y)
x = data_aran_vh['Lon']
y = data_aran_vh['Lat']
plt.plot(x, y)
plt.grid(alpha=0.5)
plt.show()
# %% Align data by latitiude, longitude and bearing 

# data_fric_hh.head()
data_iri_hh.head()
# %%
