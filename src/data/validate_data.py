from .__init__ import *

from typing import DefaultDict
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from src.data.dataloader import Platoon
from torch.utils.data import DataLoader
from collections import defaultdict 
from scipy.stats import gaussian_kde

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
            aran_latlon = np.vstack((aran[:, aran.attrs['Lat']], aran[:, aran.attrs['Lon']])).T # Lat
            p79_latlon = np.vstack((p79[:, p79.attrs['Lat']], p79[:, p79.attrs['Lon']])).T # Lat
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


def distributions_of_sensors_in_gm(segments, cols=['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2', 'acc_long', 'acc_trans', 'acc_yaw', 'spd_veh', 'odo', 'rpm']):
    plt.rcParams['patch.linewidth'] = 0
    plt.rcParams['patch.edgecolor'] = 'none'
    # gm_cols_indices = list(segments['0']['2']['gm'].attrs[v] for v in list(segments['0']['2']['gm'].attrs.keys()))
    # gm_cols = list(segments['0']['2']['gm'].attrs.keys())
    # gm_cols = ['acc.xyz_0', 'acc.xyz_1', 'acc.xyz_2', 'acc_long', 'acc_trans', 'acc_yaw', 'alt', 'asr_trq_req_dyn', 'asr_trq_req_st', 'brk_trq_elec', 'brk_trq_req_dvr', 'brk_trq_req_elec', 'distance', 'f_dist', 'gps_0', 'gps_1', 'msr_trq_req', 'odo', 'rpm', 'rpm_elec', 'rpm_fl', 'rpm_fr', 'rpm_rl', 'rpm_rr', 'sb_rem_fl', 'spd', 'spd_veh', 'strg_acc', 'strg_pos', 'time', 'trac_cons', 'trip_cons', 'trip_dist', 'trip_spd_avg', 'trq_eff', 'trq_req', 'whl_prs_fl', 'whl_prs_fr', 'whl_prs_rl', 'whl_prs_rr', 'whl_trq_est', 'whl_trq_pot_ri']
    gm_cols = cols
    gm_cols_indices = list(segments['0']['2']['gm'].attrs[v] for v in gm_cols)
    segment_dict = defaultdict(lambda: [])
    for segment in tqdm(segments.values()):
        for seconds in segment.values():
            gm = seconds['gm']
            segment_dict[segment.attrs['trip_name']].append(gm[()])
    
    # Based on length of cols, find appropriate number of rows and columns for the subplots making it as square as possible
    r = int(np.ceil(len(gm_cols) ** 0.5))
    c = int(np.ceil(len(gm_cols) / r))*2
    
    fig, axes = plt.subplots(r, c, figsize=(10*2, 10))

    kdes = {gm_col: {} for gm_col in gm_cols}
    eval_points_dict = {gm_col: {} for gm_col in gm_cols}
    for car, value in segment_dict.items():
        value = np.vstack(value)
        for gm_col in gm_cols:
            kdes[gm_col][car] = {}
            eval_points_dict[gm_col][car] = {}
        # Now plot the distributions of each column in the GM data of shape (n, 42)
        for i in tqdm(range(len(axes.flat)//2), total=r*c//2):
            ax1 = axes.flat[2*i]
            val = value[:, gm_cols_indices[i]]
            # sns.histplot(val, ax=ax, label=car, bins=30, alpha=0.2)
            # Calculate KDE of val
            eval_points = np.linspace(val.min(), val.max(), 100)
            density = gaussian_kde(val)
            kdes[gm_cols[i]][car] = density
            eval_points_dict[gm_cols[i]][car] = (eval_points.min(), eval_points.max())
            y_sp = density.pdf(eval_points)
            # plot the kde
            ax1.plot(eval_points, y_sp, label=car)
            # sns.displot(value[:, i], ax=ax, kde=True, label=car)
            if car == '16006':
                ax1.set_title(gm_cols[i])
    
    for i, gm_col in (pbar := tqdm(enumerate(gm_cols), total=len(gm_cols))):
        ax2 = axes.flat[2*i+1]
        # Compute pairwise kl divergence between the distributions of the current column
        kl_divergences = np.nan * np.ones((len(segment_dict), len(segment_dict)))
        for i, (car1, kde1) in enumerate(kdes[gm_col].items()):
            car1_eval_points = eval_points_dict[gm_col][car1]
            for j, (car2, kde2) in enumerate(kdes[gm_col].items()):
                if i >= j:
                    continue
                pbar.set_description(f"KL Divergence for {gm_col}, {car1} vs {car2}")
                car2_eval_points = eval_points_dict[gm_col][car2]
                min_eval = max(car1_eval_points[0], car2_eval_points[0])
                max_eval = min(car1_eval_points[1], car2_eval_points[1])
                eval_points = np.linspace(min_eval, max_eval, 200)
                kl_divergences[i, j] = np.sum(kde1.pdf(eval_points) * np.log((kde1.pdf(eval_points) + 1e-9) / (kde2.pdf(eval_points) + 1e-9)))
        sns.heatmap(kl_divergences, ax=ax2, xticklabels=list(segment_dict.keys()), yticklabels=list(segment_dict.keys()), cmap='viridis')

    fig.suptitle(f"Distributions of GM data for all cars")
    # insert legend below all plots in figure and remove duplicates
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5,-0.01))
    # move the legend down

    plt.tight_layout()
    plt.show()
    plt.rcParams['patch.linewidth'] = 1
    plt.rcParams['patch.edgecolor'] = 'black'


import numpy as np

from scipy.interpolate import interp1d

def upsample_gyroscope(gyroscope, new_length):
    # gyroscope is a numpy array of shape (n, 3)
    # new_length is the desired length of the upsampled data

    # create an array of the current indices
    old_indices = np.linspace(0, gyroscope.shape[0]-1, gyroscope.shape[0])

    # create an array of the new indices
    new_indices = np.linspace(0, gyroscope.shape[0]-1, new_length)

    # create a new empty array for the upsampled data
    upsampled_gyroscope = np.empty((new_length, 3))

    # iterate over each dimension
    for i in range(3):
        # create an interpolation function for this dimension
        f = interp1d(old_indices, gyroscope[:, i])

        # use the interpolation function to compute the upsampled data
        upsampled_gyroscope[:, i] = f(new_indices)

    return upsampled_gyroscope


def rotate_acceleration(acceleration, gyroscope):
    # acceleration and gyroscope are numpy arrays of shape (n, 3)
    # acceleration is measured in m/s^2
    # gyroscope is measured in rad/s

    # create an empty array to store the rotated acceleration
    rotated_acceleration = np.empty_like(acceleration)

    # iterate over each acceleration and gyroscope measurement
    for i in range(acceleration.shape[0]):
        # get the current acceleration and gyroscope measurement
        acc = acceleration[i]
        gyro = gyroscope[i]
        
        # Based on the euler angles of the gopro camera, we have an intrisic rotation of the gyroscope
        # https://www.wikiwand.com/en/Rotation_matrix
        alpha = gyro[0]
        beta = gyro[1]
        gamma = gyro[2]
        # create the rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])

        # rotate the acceleration
        rotated_acceleration[i] = (Rz @ Ry @ Rx) @ acc

    return rotated_acceleration

def plot_z_acceleration_between_gm_and_gopro(segments: h5py.File, smooth_go_pro: bool = False):
    import statsmodels.api as sm
    gyro_cols = ['Gyroscope (x) [rad_s]', 'Gyroscope (y) [rad_s]', 'Gyroscope (z) [rad_s]']
    gopro_cols = ['Accelerometer (x) [m_s2]', 'Accelerometer (y) [m_s2]', 'Accelerometer (z) [m_s2]']
    
    for segment in tqdm(segments.values()):
        for i, seconds in segment.items():
            gm = seconds['gm']
            gopro = seconds['gopro']
            
            gyro = np.stack([gopro[()][:, gopro.attrs[i]] for i in gyro_cols]).T
            
            gm_z_acc = gm[()][:, gm.attrs['acc.xyz_2']]
            
            x = np.arange(gm_z_acc.shape[0])
            gopro_acc = np.stack([gopro[()][:, gopro.attrs[i]] / 9.81 for i in gopro_cols]).T
            go_pro_rotated_accelerations = rotate_acceleration(gopro_acc, gyro)            
            gopro_z_acc = go_pro_rotated_accelerations[:, 2]
            
            if smooth_go_pro:
                gopro_z_acc = sm.nonparametric.lowess(gopro_z_acc, x, frac=0.5, is_sorted=True, return_sorted=False)
            
            fig, ax = plt.subplots()
            ax.plot(x, gm_z_acc, label='GM')
            ax.plot(x, gopro_acc[:,2], label='GoPro before rotation')
            ax.plot(x, gopro_z_acc, label='GoPro after rotation')
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Z Acceleration")
            ax.legend()
            ax.set_title(f"Z Acceleration for {segment.attrs['trip_name']} at {i} seconds")
            plt.show()  



if __name__=='__main__':

    # trainset = Platoon(data_type='train', pm_windowsize=2)
    # train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # plot_kpi_vs_avg_speed(train_loader)
    # with h5py.File('data/processed/wo_kpis/segments.hdf5', 'r') as segments:
    #     distributions_of_sensors_in_gm(segments)
    with h5py.File('data/processed/w_kpis/segments.hdf5', 'r') as segments:
        # plot_number_of_reference_points_vs_avg_speed(segments)
        # plot_number_of_reference_points_vs_normalised_second_idx(segments)
        # plot_mean_lon_lat_distance_vs_normalised_second_idx(segments)
        plot_z_acceleration_between_gm_and_gopro(segments)
        
    