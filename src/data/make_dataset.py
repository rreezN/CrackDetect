
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import h5py

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
    out = ((data - bstar*rstar) - b)*r
    return out


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
            if key in parameter_dict.keys():
                data[key] = convertdata(group[key][()], parameter_dict[key])
            else:
                data[key] = group[key][()]
    return data

if __name__ == '__main__':
    autopi_hh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5')
    autopi_vh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5')
    print('Data unpacked and converted...')