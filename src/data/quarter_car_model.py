# Based on the code by Asmus Skar: https://github.com/asmusskar/RIVA/tree/main

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
    


class QuarterCarModel():
    def __init__(self, sys):
        # Load the qcm data
        sys = loadmat("data/quarter_car/pid_control.mat")
        
        # Setup time and distance
        self.dx = np.array(sys['distance'])[1] - np.array(sys['distance'])[0] # Sampling interval [m]
        self.dt = np.array(sys['time'])[1] - np.array(sys['time'])[0] # Time step [s]
        self.v = self.dx/self.dt # Velocity in [m/s]
        
        # Manipulate velocity for testing of code (only for testing, remove later)
        vvar = 0 # Variation in velocity
        fv = 0.5 # Frequency of variation
        self.vvar = vvar * np.sin(2*np.pi*fv*sys['time']) + self.v # Varying velocity in [m/s]
        
        self.dtvar = self.dx / self.vvar
        self.time = np.cumsum(self.dtvar) # Cumulative sum of time steps

        self.Zpfm = np.array(sys['inpfilt']) * 1e3 # Road profile (filtered)
        self.Zraw = np.array(sys['severity']) * 1e3
        self.Zpfm = self.Zpfm - self.Zpfm[0]
        
        # Model parameters
        self.K1 = 653
        self.K2 = 63.3
        self.C = 6
        self.U = 0.15
        
        # Initialize response
        self.Zu0 = np.zeros(len(sys['acceleration']))
        self.Zs0 = self.Zu0
        self.Zp0 = self.Zu0
        self.acc0 = self.Zu0
        
        # Generate synthetic accelerations
        self.Zp = self.Zpfm
        self.acceleration = self.qcar_acc_ms_tvar()
        
    def qcar_acc_ms_tvar(self):
        K1 = self.K1
        K2 = self.K2
        C = self.C
        U = self.U
        Zu = self.Zu0
        Zs = self.Zs0
        Zs_dotdot = self.acc0
        time = self.time
        Zp = self.Zp.flatten()

        for i in range(1, len(Zu)-1):
            # Estimate the time increment/speed for the current step
            dt = (time[i] - time[i-1]) + (time[i+1] - time[i])/2
            
            if dt < 0:
                print('ERROR: Negative time increment at index:', i)
                break
            
            # print(dt**2*K1*(Zp[i]-Zu[i]))
            # Calculate unsprung mass displacement
            Zu[i+1] = ((dt*C+2) \
                        * (dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i-1]-2*Zu[i])+2*Zs[i]-Zs[i-1]) \
                        + 2*dt**2*K2*(Zs[i]-Zu[i])+dt*C*(Zu[i-1]-Zs[i-1])+2*Zs[i-1]-4*Zs[i]) \
                        / (dt*C*(1+U)+2*U)
            
            # Calculate sprung mass displacement
            Zs[i+1] = dt**2*K1*(Zp[i]-Zu[i])-U*(Zu[i+1]-2*Zu[i]+Zu[i-1])+2*Zs[i]-Zs[i-1]
            
            # Calculate sprung mass acceleration
            Zs_dotdot[i] = (Zs[i+1]-2*Zs[i]+Zs[i-1])/dt**2
            
        return Zs_dotdot
        


if __name__ == "__main__":
    sys = loadmat("data/quarter_car/pid_control.mat")
    model = QuarterCarModel(sys)
    plt.plot(model.acceleration)
    plt.show()
    # acc = qcar_model(sys)
    # print(acc)

            