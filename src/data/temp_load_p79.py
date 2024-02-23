import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import h5py
from scipy.interpolate import CubicSpline
from copy import deepcopy



def load_p79():
    parameter_dict = {
            'acc_long':     {'bstar': 198,      'rstar': 1,     'b': 198,   'r': 0.05   },
            'acc_trans':    {'bstar': 32768,    'rstar': 1,     'b': 32768, 'r': 0.04   },
            'acc_yaw':      {'bstar': 2047,     'rstar': 1,     'b': 2047,  'r': 0.1    },
            'brk_trq_elec': {'bstar': 4096,     'rstar': -1,    'b': 4098,  'r': -1     },
            'whl_trq_est':  {'bstar': 12800,    'rstar': 0.5,   'b': 12700, 'r': 1      },
            'trac_cons':    {'bstar': 80,       'rstar': 1,     'b': 79,    'r': 1      },
            'trip_cons':    {'bstar': 0,        'rstar': 0.1,   'b': 0,     'r': 1      }
        }

    def convertdata(data, parameter):
        bstar = parameter['bstar']
        rstar = parameter['rstar']
        b = parameter['b']
        r = parameter['r']
        # We only convert data in the second column at idx 1 (wrt. 0-indexing), as the first column is time
        col0 = data[:,0]
        col1 = ((data[:,1]-bstar*rstar)-b)*r
        data = np.column_stack((col0, col1))
        return data


    def unpack_hdf5(hdf5_file):
        with h5py.File(hdf5_file, 'r') as f:
            data = unpack_hdf5_(f)
        return data

    def unpack_hdf5_(group):
        data = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = unpack_hdf5_(group[key])
            else:
                if key in parameter_dict:
                    data[key] = convertdata(group[key][()], parameter_dict[key])
                else:
                    data[key] = group[key][()]
        return data


    p79_hh = pd.read_csv('../../data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape')
    p79_vh = pd.read_csv('../../data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')
    autopi_hh = unpack_hdf5('../../data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5')
    autopi_vh = unpack_hdf5('../../data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5')


    def find_best_start_and_end_indeces(trip: np.ndarray, section: np.ndarray, kind="l1"):
        # Find the start and end indeces of the section data that are closest to the trip data
        lon_a, lat_a = trip[:,0], trip[:,1]
        lon_b, lat_b = section[:,0], section[:,1]
        if kind == "l1":
            start_index = np.argmin(np.abs(lon_a - lon_b[0]) + np.abs(lat_a - lat_b[0]))
            end_index = np.argmin(np.abs(lon_a - lon_b[-1]) + np.abs(lat_a - lat_b[-1]))
        elif kind == "l2":
            start_index = np.argmin(np.sqrt((lon_a - lon_b[0])**2 + (lat_a - lat_b[0])**2))
            end_index = np.argmin(np.sqrt((lon_a - lon_b[-1])**2 + (lat_a - lat_b[-1])**2))

        return start_index, end_index

    def interpolate(x: np.ndarray, y: np.ndarray, kind="cubic"):
        # Interpolate data to match the time
        if kind == 'cubic':
            f = CubicSpline(x, y)
        else:
            raise ValueError(f"Interpolation method {kind} not supported")
        return f

    def cut_dataframe_by_indeces(df, start, end):
        return df.iloc[start:end]


    cut_p79_hh = cut_dataframe_by_indeces(
        p79_hh, *find_best_start_and_end_indeces(
            p79_hh[["Lon", "Lat"]].values,
            autopi_hh["p79"]['trip_1']['pass_1']["GPS"][:, ::-1]
        )
    )
    cut_p79_vh = cut_dataframe_by_indeces(
        p79_vh, *find_best_start_and_end_indeces(
            p79_vh[["Lon", "Lat"]].values,
            autopi_vh["p79"]['trip_1']['pass_1']["GPS"][:, ::-1]
        )
    )


    class InterpolationClass:
        def interpolate(self, x: np.ndarray, y: np.ndarray, kind="cubic"):
            # Interpolate data to match the time
            if kind == 'cubic':
                f = CubicSpline(x, y)
            else:
                raise ValueError(f"Interpolation method {kind} not supported")
            return f

        def normalise(self, column):
            return (column - column.min()) / (column.max() - column.min())
        
        def start_from_zero(self, column):
            return column - column.min()

    class InterpolateP79(InterpolationClass):
        laser_columns = [f" Laser {i} [mm]" for i in range(1, 26)]

        def __init__(self, df: pd.DataFrame, kind="cubic"):
            self.df = df
            self.kind = kind
            self.accumulated_distance = self.start_from_zero(df["Distance [m]"])
            
            self.lasers = self.interpolate(self.accumulated_distance, df[self.laser_columns].iloc[:, ], kind=kind)
            self.lon = self.interpolate(self.accumulated_distance, df[["Lon"]], kind=kind)
            self.lat = self.interpolate(self.accumulated_distance, df[["Lat"]], kind=kind)
            
    # Interpolate P79 laser data to match with the corresponding distance measurements
    p79_hh_interpolated = InterpolateP79(deepcopy(cut_p79_hh))
    return p79_hh_interpolated