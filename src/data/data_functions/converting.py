#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import h5py
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path
from typing import Optional
from sympy import ShapeError
from tqdm import tqdm

from .validating import clean_int


CONVERT_PARAMETER_DICT = {
    'acc_long':     {'bstar': 198,      'rstar': 1,     'b': 198,   'r': 0.05   },
    'acc_trans':    {'bstar': 32768,    'rstar': 1,     'b': 32768, 'r': 0.04   },
    'acc_yaw':      {'bstar': 2047,     'rstar': 1,     'b': 2047,  'r': 0.1    },
    'brk_trq_elec': {'bstar': 4096,     'rstar': -1,    'b': 4098,  'r': -1     },
    'whl_trq_est':  {'bstar': 12800,    'rstar': 0.5,   'b': 12700, 'r': 1      },
    'trac_cons':    {'bstar': 80,       'rstar': 1,     'b': 79,    'r': 1      },
    'trip_cons':    {'bstar': 0,        'rstar': 0.1,   'b': 0,     'r': 1      }
}

SMOOTH_PARAMETER_DICT = {
    'acc.xyz':       {'kind': 'lowess', 'frac': [0.005, 0.005, 0.001]}, # NOTE Måske den ikke skal smoothes?
    'spd_veh':       {'kind': 'lowess', 'frac': [0.005]},
    'acc_long':      {'kind': 'lowess', 'frac': [0.005]},
    'acc_trans':     {'kind': 'lowess', 'frac': [0.005]}
}


# ========================================================================================================================
#           Convertion functions
# ========================================================================================================================

def convertdata(data: np.ndarray, parameter: dict) -> np.ndarray:
    """
    Convert the data using the parameters specified in the parameter dictionary. 
    The conversion is done according to the paper,

        "LiRA-CD: An open-source dataset for road condition modelling and research" 
            by Asmus Skar et al.
        (https://doi.org/10.1016/j.dib.2023.109426)

    Using the following formula,

            s = (s_{LiRA-CD} - b^{*} * r^{*}) - b) * r
    
    - CONVERT_PARAMETER_DICT corresponds to Table 2, and contains all the parameters 
        needed for the conversion.
    - s is the converted value, and corresponds to col1 in the code.
        
    Parameters
    ----------
    data : np.ndarray
        The data to convert
    parameter : dict
        The parameters to use for the conversion.

    Returns
    -------
    np.ndarray
        The converted data
    """
    # data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input value 'data' type is {type(data)}, but expected np.ndarray")
    if not len(data.shape) == 2 or not data.shape[1] == 2:
        raise ShapeError(f"Input value 'data' shape is {data.shape}, but expected (n, 2)")
    # parameter  
    if not isinstance(parameter, dict):
        raise TypeError(f"Input value 'parameter' type is {type(parameter)}, but expected dict.")
    if "bstar" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'bstar'. Check the parameter dictionary for missing values.")
    if "rstar" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'rstar'. Check the parameter dictionary for missing values.")
    if "b" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'b'. Check the parameter dictionary for missing values.")
    if "r" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'r'. Check the parameter dictionary for missing values.")
    
    # We extract the parameters from the parameter dictionary
    bstar = parameter['bstar']
    rstar = parameter['rstar']
    b = parameter['b']
    r = parameter['r']
    # We only convert data in the second column at idx 1 (wrt. 0-indexing), as the first column is time
    col0 = data[:,0]
    col1 = ((data[:,1]-bstar*rstar)-b)*r
    data = np.column_stack((col0, col1))
    return data

def smoothdata(data: np.ndarray, parameter: dict) -> np.ndarray:
    """
    Smooth the data using the parameters specified in the parameter dictionary.
    
    To account for noisy inputs in the data, we smooth the data using the LOWESS method. 
    The smoothing was done in super-vision by Asmus Skar.
    
    Parameters
    ----------
    data : np.ndarray
        The data to convert
    parameter : dict
        The parameters to use for the conversion.

    Returns
    -------
    np.ndarray
        The smoothed data
    """
    
    # We assert that the input data is as expected
    # data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input value 'data' type is {type(data)}, but expected np.ndarray")
    # parameter
    if not isinstance(parameter, dict):
        raise TypeError(f"Input value 'parameter' type is {type(parameter)}, but expected dict.")
    if "kind" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'kind'. Check the parameter dictionary for missing values.")
    if "frac" not in parameter:
        raise KeyError(f"Input value 'parameter' does not contain 'frac'. Check the parameter dictionary for missing values.")
    if not len(parameter['frac']) == data.shape[1]-1:
        raise ValueError(f"Input value 'parameter', 'frac' length is not equal to the number of columns in the data (excluding the time column).")

    # We only smooth data in the second column at idx 1 (wrt. 0-indexing), as the first column is time
    x = data[:,0]
    kind = parameter["kind"]
    frac = parameter["frac"]
    for i in range(1, data.shape[1]):
        if kind == "lowess":
            data[:,i] = sm.nonparametric.lowess(data[:,i], x, frac=frac[i-1], is_sorted=True, return_sorted=False)
        else:
            raise NotImplementedError(f"Smoothing method {kind} not implemented")
    return data


def convert_autopi_can(original_file: h5py.Group, converted_file: h5py.Group, verbose: bool = False, pbar: Optional[tqdm] = None) -> None:
    """
    Convert the data in the original data file containing the AutoPi and CAN data and save it into the converted file.
    We create this extra step for easier handling of the data in the future, and faster troubleshooting by saving each 
    step done on the data seperately.
    
    Parameters
    ----------
    original_file : h5py.Group
        The original data file containing the AutoPi and CAN data
    converted_file : h5py.Group
        The converted data file where the converted data is saved
    verbose : bool
        Whether to show a progress bar during conversion
    pbar : Optional[tqdm]
        The progress bar to use
    """
    # original_file
    if not isinstance(original_file, h5py.Group) and not isinstance(original_file, h5py.File):
        raise TypeError(f"Input value 'original_file' type is {type(original_file)}, but expected h5py.Group or h5py.File.")
    # converted_file
    if not isinstance(converted_file, h5py.Group) and not isinstance(converted_file, h5py.File):
        raise TypeError(f"Input value 'converted_file' type is {type(converted_file)}, but expected h5py.Group or h5py.File.")
    # verbose
    if not isinstance(verbose, bool):
        raise TypeError(f"Input value 'verbose' type is {type(verbose)}, but expected bool")
    # pbar
    if pbar is not None:
        if not isinstance(pbar, tqdm):
            raise TypeError(f"Input value 'pbar' type is {type(pbar)}, but expected tqdm.")    
    
    # Specify iterator based on verbose
    if verbose:
        pbar = tqdm(total=get_total_subgroups(original_file) + 1)
    iterator = original_file.keys()

    # Convert the data in the original AutoPi CAN file to the converted file
    for key in iterator:
        if pbar is not None:
            pbar.set_description(f"Converting {original_file.name}/{key}")

        # Traverse the hdf5 tree
        if isinstance(original_file[key], h5py.Group):
            subgroup = converted_file.create_group(key)
            convert_autopi_can(original_file[key], subgroup, pbar=pbar)

        # Convert the data
        else:
            data = original_file[key][()]
            if key in CONVERT_PARAMETER_DICT:
                data = convertdata(data, CONVERT_PARAMETER_DICT[key])
            if key in SMOOTH_PARAMETER_DICT:
                data = smoothdata(data, SMOOTH_PARAMETER_DICT[key])
            
            # Save the data to the converted file
            converted_file.create_dataset(key, data=data)

    if pbar is not None:
        pbar.update(1)

def reorient_autopi_can(converted_file: h5py.Group) -> None:
    """
    Reorient the acceleration sensors in the converted AutoPi and CAN data.

    Parameters
    ----------
    converted_file : h5py.Group
        The converted AutoPi and CAN data to reorient
    """
    if not isinstance(converted_file, h5py.Group) and not isinstance(converted_file, h5py.File):
        raise TypeError(f"Input value 'converted_file' type is {type(converted_file)}, but expected h5py.Group or h5py.File.")
    
    # Go through all the trips and passes in the converted file
    for trip_name, trip in (pbar := tqdm(converted_file['GM'].items())):
        for pass_name, pass_ in trip.items():
            pbar.set_description(f"Orienting: {pass_.name}")
            reorient_pass(pass_)


def reorient_pass(pass_group: h5py.Group) -> None:
    """
    Reorient the acceleration sensors in the pass, and update the pass group in place.
    The reorientation is done based on the correlation between the CAN accelerations and the AutoPi accelerations.

    Parameters
    ----------
    pass_group : h5py.Group
        The pass group containing the converted AutoPi and CAN data to reorient
    """
    if not isinstance(pass_group, h5py.Group) and not isinstance(pass_group, h5py.File):
        raise TypeError(f"Input value 'pass_group' type is {type(pass_group)}, but expected h5py.Group or h5py.File.")
    
    # Create custom warn message (Used to tell the user that the sensors are reoriented without interrupting tqdm progress bar)
    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return '\n' + str(msg) + '\n'

    warnings.formatwarning = custom_formatwarning
    
    fs = 10 # Sampling frequency

    # Speed distance
    tspd = pass_group['spd_veh'][:, 0]
    dspd = np.cumsum(pass_group['spd_veh'][1:, 1]*np.diff(pass_group['spd_veh'][:, 0]))/3.6
    dspd = np.insert(dspd, 0, 0)

    # GPS data
    tgps = pass_group['gps'][:, 0]

    # Normalize accelerations
    taccrpi = pass_group['acc.xyz'][:, 0]
    xaccrpi = pass_group['acc.xyz'][:, 1] - np.mean(pass_group['acc.xyz'][:, 1])
    yaccrpi = pass_group['acc.xyz'][:, 2] - np.mean(pass_group['acc.xyz'][:, 2])

    tatra = pass_group['acc_trans'][:, 0]
    atra = pass_group['acc_trans'][:, 1] - np.mean(pass_group['acc_trans'][:, 1])
    talon = pass_group['acc_long'][:, 0]
    alon = pass_group['acc_long'][:, 1] - np.mean(pass_group['acc_long'][:, 1])

    # Resample to 100Hz
    time_start_max = np.max([taccrpi[0], tatra[0], talon[0], tgps[0], tspd[0]])
    time_end_min = np.min([taccrpi[-1], tatra[-1], talon[-1], tgps[-1], tspd[-1]])
    tend = time_end_min - time_start_max
    time = np.arange(0, tend, 1/fs)

    # Interpolate
    axrpi_100hz = clean_int(taccrpi-time_start_max, xaccrpi, time)
    ayrpi_100hz = clean_int(taccrpi-time_start_max, yaccrpi, time)
    aycan_100hz = clean_int(tatra-time_start_max, atra, time)
    axcan_100hz = clean_int(talon-time_start_max, alon, time)

    # Reorient accelerations
    alon = axcan_100hz.copy()
    atrans = aycan_100hz.copy()
    axpn = axrpi_100hz * 9.81
    aypn = ayrpi_100hz * 9.81

    # Calculate correlation with CAN accelerations
    pcxl = np.corrcoef(axpn, alon)[0, 1]
    pcyl = np.corrcoef(aypn, alon)[0, 1]

    pcxt = np.corrcoef(axpn, atrans)[0, 1]
    pcyt = np.corrcoef(aypn, atrans)[0, 1]
    
    # Determine the orientation of the sensors
    # NOTE: Here we also alter the entries in the original converted data if necessary!
    if (abs(pcxl) < abs(pcxt)) and (abs(pcyl) > abs(pcyt)):
        if pcxt < 0:
            warnings.warn("NOTE: Reorienting the autopi acceleration sensors as (x, y, z) -> (y, -x, z)")
            y_acc = pass_group['acc.xyz'][:, 2]                    
            pass_group['acc.xyz'][:, 2] = -pass_group['acc.xyz'][:, 1]    # y = -x
            pass_group['acc.xyz'][:, 1] = y_acc                    # x = y

        else:
            warnings.warn("NOTE: Reorienting the autopi acceleration sensors as (x, y, z) -> (y, x, z)")
            y_acc = pass_group['acc.xyz'][:, 2]
            pass_group['acc.xyz'][:, 2] = pass_group['acc.xyz'][:, 1]    # y = x
            pass_group['acc.xyz'][:, 1] = y_acc                   # x = y

    else:
        if pcyt < 0:
            warnings.warn("NOTE: Reorienting the autopi acceleration sensors as (x, y, z) -> (x, -y, z)")
            pass_group['acc.xyz'][:, 2] = -pass_group['acc.xyz'][:, 2]    # y = -y
        else:
            # No reorientation needed
            pass


def get_total_subgroups(group: h5py.Group) -> int:
    """
    Get the total number of subgroups in the group.

    Parameters
    ----------
    group : h5py.Group
        The group to get the total number of subgroups from

    Returns
    -------
    int
        The total number of subgroups in the group
    """
    if not isinstance(group, h5py.Group) and not isinstance(group, h5py.File):
        raise TypeError(f"Input value 'group' type is {type(group)}, but expected h5py.Group or h5py.File.")
    
    sub_groups = 0
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            sub_groups += 1
            sub_groups += get_total_subgroups(group[key])
    return sub_groups


def convert(hh: str = 'data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5', vh: str = 'data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5') -> None:
    """
    Main function for converting the AutoPi CAN data to the converted data.
    It loads data from a specific path, converts the data, and saves it to a new path.

    This function assumes/does the following:
        - The data is stored isn hdf5 format
        - Saves the converted data in hdf5 format at 'data/interim/gm'

    Parameters
    ----------
    hh : str
        The path to the AutoPi CAN data for the HH direction
    vh : str
        The path to the AutoPi CAN data for the VH direction
    """
    
    # hh
    if not isinstance(hh, str):
        raise TypeError(f"Input value 'hh' type is {type(hh)}, but expected str.")
    if not Path(hh).exists():
        raise FileNotFoundError(f"Path '{hh}' does not exist.")
    # vh
    if not isinstance(vh, str):
        raise TypeError(f"Input value 'vh' type is {type(vh)}, but expected str.")
    if not Path(vh).exists():
        raise FileNotFoundError(f"Path '{vh}' does not exist.")
    
    prefix = hh.split("data/")[0]

    hh = Path(hh)
    vh = Path(vh)

    interim_gm = Path(prefix + 'data/interim/gm')
    interim_gm.mkdir(parents=True, exist_ok=True)


    for file in [hh, vh]:
        with h5py.File(file, 'r') as f:
            with h5py.File(interim_gm / f"converted_{file.name}", 'w') as converted_file:
                convert_autopi_can(f, converted_file, verbose=True)
                reorient_autopi_can(converted_file)