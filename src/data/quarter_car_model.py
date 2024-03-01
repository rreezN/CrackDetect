# Based on the code by Asmus Skar: https://github.com/asmusskar/RIVA/tree/main

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


class QuarterCarModel:
    def __init__(self, segment, lasers=[5, 21]):
        self.segment = segment
        self.lasers = lasers
        
        # Initialize model parameters
        self.K1 = 318.61    # [s^-2]
        self.K2 = 132.32    # [s^-2]
        self.C = 5.68       # [s^-1]
        self.U = 0.09
        
        # Load and initialize p79 data
        self.Zraw, self.p79_distance = self.load_raw_profile()
        self.Zpfm = self.moving_average()
        
        # Load green mobility data
        self.gm_acc, self.gm_velocity = self.load_gm()
        
        # Calculate time interval
        self.dx = self.p79_distance[1] - self.p79_distance[0] # sampling interval [m]
        self.dt = self.dx / self.gm_velocity
        self.time = np.cumsum(self.dt)
        
        # Initialize respones
        self.Zu0 = np.zeros(len(self.p79_distance))
        self.Zs0 = np.zeros(len(self.p79_distance))
        self.Zp0 = np.zeros(len(self.p79_distance))
        self.acc0 = np.zeros(len(self.p79_distance))
        
        # Get synthethic acceleration
        self.Zp = self.Zpfm.copy()
        self.synth_acc = self.get_synth_acc()
        
    
    def load_gm(self):
        """Load green mobility data from file

        Returns:
            numpy array, numpy_array: Green Mobility z-acceleration data, Green mobility velocity data
        """
        data_path = f'data/processed/'
        gm_df = pd.read_csv(data_path + f'gm/{self.segment}_gm.csv')
        # TODO: Maybe fix the column names ?
        gm_z_acc = gm_df['acc.z'].to_numpy()
        gm_velocity = gm_df['spd_veh'].to_numpy()
        
        return gm_z_acc, gm_velocity
    
    def load_raw_profile(self):
        """Load raw profile data from file, average the two laser measurements and convert to m

        Returns:
            numpy array, numpy array: Raw road profile data, distance data
        """
        data_path = f'data/processed/'
        profile_df = pd.read_csv(data_path + f'p79/{self.segment}_p79.csv')
        profile = ((profile_df[f'Laser {self.lasers[0]} [mm]'].to_numpy() + profile_df[f'Laser {self.lasers[1]} [mm]'].to_numpy()) * 1e-3) / 2
        distance = profile_df['Distance [m]'].to_numpy()
        
        return profile, distance
    
    def moving_average(self, window_size=5):
        """Calculate the moving average of a profile, to smoothen the data according to wheel length

        Args:
            profile (Raw road profile data): The raw road profile data (from load_raw_profile())
            window_size (int, optional): The window size, corresponding to wheel length. Defaults to 5.

        Returns:
            Smoothened road profile: Smoothened road profile data
        """
        return np.convolve(self.Zraw, np.ones(window_size)/window_size, mode='same')
    
    def get_synth_acc(self):
        """Calculate synthetic acceleration from road profile

        Returns:
            numpy array: Synthetic acceleration
        """
        K1 = self.K1
        K2 = self.K2
        C = self.C
        U = self.U
        Zu = self.Zu0.copy()
        Zs = self.Zs0.copy()
        Zs_dotdot = self.acc0.copy()
        time = self.time.copy()
        Zp = self.Zp.copy().flatten()
        
        for i in range(1, len(Zu)-1):
            # Estimate the time increment/speed for the current step
            dt = (time[i] - time[i-1]) + (time[i+1] - time[i])/2
            
            if dt < 0:
                print('ERROR: Negative time increment at index:', i)
                break
            
            # Calculate unsprung mass displacement
            Zu[i+1] = ((dt*C+2) \
                        * ((dt**2)*K1*(Zp[i]-Zu[i])-U*(Zu[i-1]-2*Zu[i])+2*Zs[i]-Zs[i-1]) \
                        + 2*(dt**2)*K2*(Zs[i]-Zu[i])+dt*C*(Zu[i-1]-Zs[i-1]) \
                        + 2*Zs[i-1]-4*Zs[i]) \
                        / (dt*C*(1+U)+2*U)
            
            # Calculate sprung mass displacement
            Zs[i+1] = (dt**2)*K1*(Zp[i]-Zu[i])-U*(Zu[i+1]-2*Zu[i]+Zu[i-1])+2*Zs[i]-Zs[i-1]
            
            # Calculate sprung mass acceleration
            Zs_dotdot[i] = (Zs[i+1]-2*Zs[i]+Zs[i-1])/(dt**2)
            
        return Zs_dotdot
    
    def get_euclidean_distance(self):
        """Calculate the euclidean distance between the synthetic acceleration and the green mobility acceleration

        Returns:
            float: Euclidean distance
        """
        return np.linalg.norm(self.synth_acc - self.gm_acc)
    
    def get_cross_correlation(self):
        """Calculate the cross-correlation between the synthetic acceleration and the green mobility acceleration

        Returns:
            float: Cross-correlation
        """
        return np.correlate(self.synth_acc, self.gm_acc)
    
    def plot_road_profile(self):
        """Plot the road profile
        """
        plt.figure(figsize=(15,5))
        plt.plot(self.p79_distance, self.Zraw, color='black', label='Raw profile', linewidth=0.5)
        plt.plot(self.p79_distance, self.Zpfm, color='red', label='Smoothened profile', linestyle='dotted')
        plt.title(f'{self.segment} road profile')
        plt.xlabel('Distance [m]')
        plt.ylabel('Elevation [mm]')
        plt.tight_layout()
        plt.savefig(f'reports/figures/quarter_car_model/{self.segment}_road_profile.png', dpi=300)
        plt.show()    
        
    def plot_synth_acc(self):
        """Plot the synthetic acceleration
        """
        plt.figure(figsize=(15,5))
        plt.plot(self.p79_distance, self.gm_acc, color='black', label='Green Mobility acceleration', linewidth=0.5)
        plt.plot(self.p79_distance, self.synth_acc, color='red', label='Synthetic acceleration', linestyle='dotted')
        plt.title(f'{self.segment} synthetic acceleration')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.tight_layout()
        plt.savefig(f'reports/figures/quarter_car_model/{self.segment}_synth_acc.png', dpi=300)
        plt.show()
    

def argparser():
    parser = argparse.ArgumentParser(prog="Quarter Car Model", description='Creates a quarter car model and compares it to green mobility data')
    parser.add_argument('--segment', type=str, default='segment_001', help='The segment to analyze')
    parser.add_argument('--lasers', type=list, default=[5, 21], help='The lasers to use for the road profile')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    qcm = QuarterCarModel(args.segment, args.lasers)
    qcm.plot_road_profile()
    qcm.plot_synth_acc()
    print(f'Euclidean distance: {qcm.get_euclidean_distance()}')
    print(f'Cross-correlation: {qcm.get_cross_correlation()}')

