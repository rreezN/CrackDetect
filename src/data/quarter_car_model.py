# Based on the code by Asmus Skar: https://github.com/asmusskar/RIVA/tree/main

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
import pandas as pd



class QuarterCarModel:
    def __init__(self, raw_profile=True, lasers=[5, 21], get_acc=True, get_profile=False, get_pid=False):
        self.raw_profile = raw_profile
        self.get_acc = get_acc
        self.get_profile = get_profile
        self.get_pid = get_pid
        
        # Model parameters
        self.K1 = 653
        self.K2 = 63.3
        self.C = 6
        self.U = 0.15
        
        if raw_profile is not None:
            from temp_load_p79 import load_p79
            # Temporary load of p79 data
            self.Zinterpolated = load_p79()
            distance = self.Zinterpolated.df['Distance [m]'].to_numpy()
            self.dx = distance[1] - distance[0] # Sampling interval [m]
            self.dt = 0.01 # Time step [s] --> THIS IS A GUESS
            self.v = self.dx/self.dt # Velocity in [m/s]
            
            # Manipulate velocity for testing of code (only for testing, remove later)
            x = np.ones(len(distance-1))
            vvar = 0 # Variation in velocity
            fv = 0.5 # Frequency of variation
            self.vvar = vvar * np.sin(2*np.pi*fv*distance) + self.v # Varying velocity in [m/s]
            
            self.dtvar = self.dx / self.vvar
            self.time = np.cumsum(self.dtvar) # Cumulative sum of time steps
            
            laser1 = self.Zinterpolated.df[f" Laser {lasers[0]} [mm]"].to_numpy()
            laser2 = self.Zinterpolated.df[f" Laser {lasers[1]} [mm]"].to_numpy()
            
            self.Zraw = self.average_profile(laser1, laser2) # Road profile
            self.Zpfm = self.moving_average(self.Zraw) # Road profile (filtered)
            # self.Zpfm = self.Zpfm - self.Zpfm[0]
            
            # Initialize response
            self.Zu0 = np.zeros(len(distance))
            self.Zs0 = self.Zu0.copy()
            self.Zp0 = self.Zu0.copy()
            self.acc0 = self.Zu0.copy()
            
        
        else:
            # Load the qcm data
            self.sys = loadmat("../../data/quarter_car/pid_control.mat") # interactive window
            # self.sys = loadmat("data/quarter_car/pid_control.mat") # non-interactive window
            
            # Setup time and distance
            self.dx = np.array(self.sys['distance'])[1] - np.array(self.sys['distance'])[0] # Sampling interval [m]
            self.dt = np.array(self.sys['time'])[1] - np.array(self.sys['time'])[0] # Time step [s]
            self.v = self.dx/self.dt # Velocity in [m/s]
            
            # Manipulate velocity for testing of code (only for testing, remove later)
            x = np.ones(len(self.sys['time']-1))
            vvar = 0 # Variation in velocity
            fv = 0.5 # Frequency of variation
            self.vvar = vvar * np.sin(2*np.pi*fv*self.sys['time']) + self.v # Varying velocity in [m/s]
            
            self.dtvar = self.dx / self.vvar
            self.time = np.cumsum(self.dtvar) # Cumulative sum of time steps

            # Asmus multiplies by 1000, not sure why, gives wrong scale compared to the figures from the paper, and he reconverts to 1e-3 in the plots
            self.Zpfm = np.array(self.sys['inpfilt']) * 1e3 # Road profile (filtered)
            self.Zraw = np.array(self.sys['severity']) * 1e3
            self.Zpfm = self.Zpfm - self.Zpfm[0]
            
            # Initialize response
            self.Zu0 = np.zeros(len(self.sys['acceleration']))
            self.Zs0 = self.Zu0.copy()
            self.Zp0 = self.Zu0.copy()
            self.acc0 = self.Zu0.copy()

        # -----------------------------------------------------
        # Generate synthetic accelerations
        # -----------------------------------------------------
        if self.get_acc:
            self.Zp = self.Zpfm.copy()
            self.accm = self.qcar_acc_ms_tvar()
        
        
        # Im not sure if we need the rest of this, if we just want to use the synthetic acceleration
        # -----------------------------------------------------
        # Optimization options
        # -----------------------------------------------------
        if self.get_pid:
            # Set solver
            self.solver = 'nms' # 'nms', 'grad', 'lsq'
            
            # Setup variable parameters
            Kp0 = 0
            Ki0 = 0
            Kd0 = 0
            X0 = [Kp0, Ki0, Kd0]
            
            # -----------------------------------------------------
            # Run optimization algorithm
            # -----------------------------------------------------
            options = {'maxiter': 1e6}
            result = minimize(self.PID_profile_inv_tvar, X0, method='Nelder-Mead', options=options)
            xmin = result.x # Optimal solution
            fval = result.fun # Function value at the optimal solution
            exitflag = result.status # Exit flag (0 for succes, 1 for failure)
        
        # -----------------------------------------------------
        # Post processing
        # -----------------------------------------------------
        if self.get_profile:
            # Get optimal profile
            self.output, self.pinv, self.accinv = self.PID_profile_out_tvar(xmin)
    
    def average_profile(self, laser1, laser2):
        return ((laser1 + laser2) * 1e-3) / 2
    
    def moving_average(self, profile, n=5) :
        return np.convolve(profile, np.ones(n)/n, 'same')
    
    def qcar_acc_ms_tvar(self):
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
    
    def PID_profile_inv_tvar(self, X):
        # identify variable parameters
        Kp = X[0]
        Ki = X[1]
        Kd = X[2]
        
        # Constant input parameters
        Zsm_dotdot = self.accm.copy()
        Zs_dotdot = self.acc0.copy()
        Zp = self.Zp0.copy()
        Zu = self.Zu0.copy()
        Zs = self.Zs0.copy()
        K1 = self.K1
        K2 = self.K2
        C = self.C
        U = self.U
        time = self.time.copy()
        
        # initialize error terms
        ep = self.acc0.copy()
        ei = self.acc0.copy()
        ed = self.acc0.copy()
        
        n = len(Zsm_dotdot)
        for i in range(1, n-1):
            # Estimate the time incremenet/speed for the current step
            dt = (time[i] - time[i-1]) + (time[i+1] - time[i])/2
            if dt < 0:
                print('ERROR: Negative time increment at index:', i)
                break
    
            # Calculate unsprung mass displacement
            Zu[i+1] = ((dt*C+2) \
                        * (dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i-1]-2*Zu[i])+2*Zs[i]-Zs[i-1]) \
                        + 2*dt**2*K2*(Zs[i]-Zu[i])+dt*C*(Zu[i-1]-Zs[i-1]+2*Zs[i-1]-4*Zs[i])) \
                        / (dt*C*(1+U)+2*U)
            
            # Calculate sprung mass displacement
            Zs[i+1] = dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i+1]-2*Zu[i]+Zu[i-1])+2*Zs[i]-Zs[i-1]
            
            # Calculate sprung mass acceleration
            Zs_dotdot[i] = (Zs[i+1]-2*Zs[i]+Zs[i-1])/(dt**2)
            
            # Error terms
            ep[i] = Zsm_dotdot[i] - Zs_dotdot[i]
            ei[i] = 0.5*dt*(ep[i]+ep[i-1])+ei[i-1]
            ed[i] = (Zsm_dotdot[i+1]-Zsm_dotdot[i-1])/(2*dt)-(Zs_dotdot[i]-Zs_dotdot[i-1])/dt
            
            # Calculate profile
            Zp[i+1]= Kp*ep[i]+Ki*ei[i]+Kd*ed[i]
            
            # Update error terms for summation
            ei[i-1] = ei[i]
                
        # if np.shape(Zsm_dotdot)[1] > np.shape(Zs_dotdot)[1]:
        #     Zsm_dotdot = np.transpose(Zsm_dotdot)
        # elif np.shape(Zsm_dotdot)[1] < np.shape(Zs_dotdot)[1]:
        #     Zsm_dotdot = np.transpose(Zsm_dotdot)
            
        # Calculate residual
        residual = Zsm_dotdot - Zs_dotdot
        
        # Select solver type
        if self.solver == 'nms':
            output = sum(abs(residual))
        elif self.solver == 'grad':
            output = sum(abs(residual))
        elif self.solver == 'lsq':
            output = residual
        
        return output
    
    def PID_profile_out_tvar(self, X):
        # identify variable parameters
        Kp = X[0]
        Ki = X[1]
        Kd = X[2]
        
        # Constant input parameters
        Zsm_dotdot = self.accm.copy()
        Zs_dotdot = self.acc0.copy()
        Zp = self.Zp0.copy()
        Zu = self.Zu0.copy()
        Zs = self.Zs0.copy()
        K1 = self.K1
        K2 = self.K2
        C = self.C
        U = self.U
        time = self.time.copy()
        
        # initialize error terms
        ep = self.acc0.copy()
        ei = self.acc0.copy()
        ed = self.acc0.copy()
        
        n = len(Zsm_dotdot)
        for i in range(1, n-1):
            # Estimate the time incremenet/speed for the current step
            dt = (time[i] - time[i-1]) + (time[i+1] - time[i])/2
            if dt < 0:
                print('ERROR: Negative time increment at index:', i)
                break
            
            # Calculate unsprung mass displacement
            Zu[i+1] = ((dt*C+2) \
                        * (dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i-1]-2*Zu[i])+2*Zs[i]-Zs[i-1]) \
                        + 2*dt**2*K2*(Zs[i]-Zu[i])+dt*C*(Zu[i-1]-Zs[i-1]+2*Zs[i-1]-4*Zs[i])) \
                        / (dt*C*(1+U)+2*U)
            
            # Calculate sprung mass displacement
            Zs[i+1] = dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i+1]-2*Zu[i]+Zu[i-1])+2*Zs[i]-Zs[i-1]
            
            # Calculate sprung mass acceleration
            Zs_dotdot[i] = (Zs[i+1]-2*Zs[i]+Zs[i-1])/(dt**2)
            
            # Error terms
            ep[i] = Zsm_dotdot[i] - Zs_dotdot[i]
            ei[i] = 0.5*dt*(ep[i]+ep[i-1])+ei[i-1]
            ed[i] = (Zsm_dotdot[i+1]-Zsm_dotdot[i-1])/(2*dt)-(Zs_dotdot[i]-Zs_dotdot[i-1])/dt
            
            # Calculate profile
            Zp[i+1]= Kp*ep[i]+Ki*ei[i]+Kd*ed[i]
            
            # Update error terms for summation
            ei[i-1] = ei[i]
        
        # Calculate residual
        residual = Zsm_dotdot - Zs_dotdot
        
        # Select solver type
        if self.solver == 'nms':
            output = sum(abs(residual))
        elif self.solver == 'grad':
            output = sum(abs(residual))
        elif self.solver == 'lsq':
            output = residual
        
        return output, Zp, Zs_dotdot


if __name__ == "__main__":
    model = QuarterCarModel()
    
    if model.raw_profile is not None:
        dist = model.Zinterpolated.df['Distance [m]'].to_numpy()
        dist = dist - dist[0]
        raw_profile = model.Zraw*1e3
        filtered_profile = model.Zpfm*1e3
    else:
        dist = model.sys['distance']
        raw_profile = model.sys['severity']
        filtered_profile = model.sys['inpfilt']
    
    # Plot synthetic profile of road
    plt.figure(figsize=(15, 5))
    plt.title('Road Profile')
    plt.plot(dist, raw_profile, color='black', label='Raw profile', linewidth=0.5)
    plt.plot(dist, filtered_profile, color='blue', linestyle='dotted', label='Filtered profile')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [mm]')
    plt.legend()
    plt.show()
    
    # Plot profile inversion
    if model.get_profile and model.get_pid:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        axs = axs.flat
        
        axs[0].plot(dist, model.Zraw, color='red', linestyle='--', linewidth=1.5, label='True')
        axs[0].plot(dist, model.Zpfm, color='blue', linestyle='dotted', linewidth=1.2, label='Calculated')
        axs[0].legend()
        axs[0].set_ylabel('Elevation [mm]')
        axs[0].set_title('Profile inversion')
        
        axs[1].set_title('Acceleration')
        axs[1].plot(dist[10:], model.accm[10:]*1e-3, color='red', linestyle='--', linewidth=1.5, label='True')
        axs[1].plot(dist[10:], model.pinv[10:]*1e-3, color='blue', linestyle='dotted', linewidth=1.2, label='Calculated')
        axs[1].set_ylabel('Acceleration [m/s^2]')
    
        plt.xlabel('Distance [m]')
        plt.tight_layout()
        plt.show()
    
    if model.get_acc:
        # Plot synthetic acceleration
        plt.figure(figsize=(15, 5))
        plt.title('Synthetic acceleration')
        plt.plot(dist, model.accm, color='red', linestyle='--')
        plt.xlabel('Distance [m]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.tight_layout()
        plt.show()
    

    
    
    

            