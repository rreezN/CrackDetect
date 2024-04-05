import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from dataloader import Platoon
from torch.utils.data import DataLoader


def plot_kpi_vs_avg_speed(train_loader: DataLoader):
    # Make plot with seaborn
    sns.set_theme(style="whitegrid")
    avg_speeds = []
    kpis = []
    for data_segment, target_segment in tqdm(train_loader):
        avg_speeds.append(data_segment.mean(axis=(1,2)))
        kpis.append(target_segment)
    avg_speeds = torch.hstack(avg_speeds)
    kpis = torch.vstack(kpis)
    kpi_names = ["Damage Index", "Rutting Index", "Patching Index", "IRI"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        sns.scatterplot(x=avg_speeds, y=kpis[:, i], ax=ax)
        ax.set_xlabel("Average Speed")
        ax.set_ylabel(kpi_names[i])
    fig.suptitle("KPI vs Average Speed")
    plt.show()


def plot_number_of_reference_points_vs_avg_speed(segments: h5py.File):
    # Make plot with seaborn
    sns.set_theme(style="whitegrid")
    avg_speeds = []
    num_aran_points = []
    num_p79_points = []
    for segment in tqdm(segments.values()):
        for seconds in segment.values():
            aran = seconds['aran']
            p79 = seconds['p79']
            gm = seconds['gm']
            num_aran_points.append(aran[()].shape[0])
            num_p79_points.append(p79[()].shape[0])
            avg_speeds.append(gm[()][:, gm.attrs['spd_veh']].mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=avg_speeds, y=num_aran_points, ax=axes[0])
    axes[0].set_xlabel("Average Speed")
    axes[0].set_ylabel("Number of Aran Points")
    sns.scatterplot(x=avg_speeds, y=num_p79_points, ax=axes[1])
    axes[1].set_xlabel("Average Speed")
    axes[1].set_ylabel("Number of P79 Points")
    fig.suptitle("Number of Reference Points vs Average Speed")
    plt.show()


def plot_number_of_reference_points_vs_normalised_second_idx(segments: h5py.File):
    # Make plot with seaborn
    sns.set_theme(style="whitegrid")
    norm_time_idx = []
    num_aran_points = []
    num_p79_points = []
    for segment in tqdm(segments.values()):
        max_index = max([int(i) for i in segment.keys()])
        for i, seconds in segment.items():
            aran = seconds['aran']
            p79 = seconds['p79']
            norm_time_idx.append(int(i) / max_index)
            num_aran_points.append(aran[()].shape[0])
            num_p79_points.append(p79[()].shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=norm_time_idx, y=num_aran_points, ax=axes[0])
    axes[0].set_xlabel("Normalised Time Index")
    axes[0].set_ylabel("Number of Aran Points")
    sns.scatterplot(x=norm_time_idx, y=num_p79_points, ax=axes[1])
    axes[1].set_xlabel("Normalised Time Index")
    axes[1].set_ylabel("Number of P79 Points")
    fig.suptitle("Number of Reference Points vs Normalised Time Index")
    plt.tight_layout()
    plt.show()

def plot_mean_lon_lat_distance_vs_normalised_second_idx(segments: h5py.File):
    # Make plot with seaborn
    sns.set_theme(style="whitegrid")
    norm_time_idx = []
    mean_lon_lat_distance_gm_aran = []
    mean_lon_lat_distance_gm_p79 = []
    for segment in tqdm(segments.values()):
        max_index = max([int(i) for i in segment.keys()])
        for i, seconds in segment.items():
            aran = seconds['aran']
            p79 = seconds['p79']
            gm = seconds['gm']
            norm_time_idx.append(int(i) / max_index)
            gm_to_aran = []
            gm_to_p79 = []
            aran_latlon = aran[()][:, aran.attrs['Lat']:aran.attrs['Lon']+1] # Lat
            p79_latlon = p79[()][:, p79.attrs['Lat']:p79.attrs['Lon']+1] # Lat
            for j in [0, gm[()].shape[0]-1]:
                gm_lat = gm[()][j, gm.attrs['gps_0']] # Lat
                gm_lon = gm[()][j, gm.attrs['gps_1']] # Lon
                gm_latlon = np.array([[gm_lat, gm_lon]])
                
                gm_to_aran.append(broadcasting_based_lng_lat_elementwise(gm_latlon, aran_latlon).mean())
                gm_to_p79.append(broadcasting_based_lng_lat_elementwise(gm_latlon, p79_latlon).mean())

            mean_lon_lat_distance_gm_aran.append(np.mean(gm_to_aran) * 1000)
            mean_lon_lat_distance_gm_p79.append(np.mean(gm_to_p79) * 1000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=norm_time_idx, y=mean_lon_lat_distance_gm_aran, ax=axes[0])
    axes[0].set_xlabel("Normalised Time Index")
    axes[0].set_ylabel("Mean Lon-Lat Distance (m) GM to Aran")
    sns.scatterplot(x=norm_time_idx, y=mean_lon_lat_distance_gm_p79, ax=axes[1])
    axes[1].set_xlabel("Normalised Time Index")
    axes[1].set_ylabel("Mean Lon-Lat Distance (m) GM to P79")
    fig.suptitle("Mean Lon-Lat Distance vs Normalised Time Index")
    plt.tight_layout()
    plt.show()

def broadcasting_based_lng_lat_elementwise(data1, data2):
    # data1, data2 are the data arrays with 2 cols and they hold
    # lat., lng. values in those cols respectively
    data1 = np.deg2rad(data1)                     
    data2 = np.deg2rad(data2)                     

    lat1 = data1[:,0]                     
    lng1 = data1[:,1]         

    lat2 = data2[:,0]                     
    lng2 = data2[:,1]         

    diff_lat = lat1 - lat2
    diff_lng = lng1 - lng2
    d = np.sin(diff_lat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(diff_lng/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

if __name__=='__main__':

    trainset = Platoon(data_type='train', pm_windowsize=2, gm_cols=['spd_veh'])
    train_loader = DataLoader(trainset, batch_size=None, shuffle=True, num_workers=0)

    # plot_kpi_vs_avg_speed(train_loader)
    
    with h5py.File('data/processed/segments.hdf5', 'r') as segments:
        plot_number_of_reference_points_vs_avg_speed(segments)
        plot_number_of_reference_points_vs_normalised_second_idx(segments)
        plot_mean_lon_lat_distance_vs_normalised_second_idx(segments)