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
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import statsmodels.api as sm


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
    'acc.xyz':       {'kind': 'lowess', 'frac': 0.005}, # NOTE Måske den ikke skal smoothes?
    'spd_veh':       {'kind': 'lowess', 'frac': 0.005},
    'acc_long':      {'kind': 'lowess', 'frac': 0.005},
    'acc_trans':     {'kind': 'lowess', 'frac': 0.005}
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
        raise TypeError(f"Data type is {type(data)}, but expected np.ndarray")
    assert len(data.shape) == 2, f"Data shape is {data.shape}, but expected (n, 2)" 
    assert data.shape[1] == 2, f"Data shape is {data.shape}, but expected (n, 2)"
    # parameter  
    if not isinstance(parameter, dict):
        raise TypeError(f"Parameter type is {type(parameter)}, but expected dict.")
    if "bstar" not in parameter:
        raise KeyError(f"Parameter does not contain 'bstar'. Check the parameter dictionary for missing values.")
    if "rstar" not in parameter:
        raise KeyError(f"Parameter does not contain 'rstar'. Check the parameter dictionary for missing values.")
    if "b" not in parameter:
        raise KeyError(f"Parameter does not contain 'b'. Check the parameter dictionary for missing values.")
    if "r" not in parameter:
        raise KeyError(f"Parameter does not contain 'r'. Check the parameter dictionary for missing values.")
    
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
        raise TypeError(f"Data type is {type(data)}, but expected np.ndarray")
    # parameter
    if not isinstance(parameter, dict):
        raise TypeError(f"Parameter type is {type(parameter)}, but expected dict.")
    if "kind" not in parameter:
        raise KeyError(f"Parameter does not contain 'kind'. Check the parameter dictionary for missing values.")
    if "frac" not in parameter:
        raise KeyError(f"Parameter does not contain 'frac'. Check the parameter dictionary for missing values.")


    # We only smooth data in the second column at idx 1 (wrt. 0-indexing), as the first column is time
    x = data[:,0]
    kind = parameter["kind"]
    frac = parameter["frac"]
    for i in range(1, data.shape[1]):
        if kind == "lowess":
            data[:,i] = sm.nonparametric.lowess(data[:,i], x, frac=frac, is_sorted=True, return_sorted=False)
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
        raise TypeError(f"Original file type is {type(original_file)}, but expected h5py.Group or h5py.File.")
    # converted_file
    if not isinstance(converted_file, h5py.Group) and not isinstance(converted_file, h5py.File):
        raise TypeError(f"Converted file type is {type(converted_file)}, but expected h5py.Group or h5py.File.")
    # verbose
    if not isinstance(verbose, bool):
        raise TypeError(f"Verbose type is {type(verbose)}, but expected bool")
    # pbar
    if pbar is not None:
        if not isinstance(pbar, tqdm):
            raise TypeError(f"Progress bar type is {type(pbar)}, but expected tqdm.")    
    
    # Specify iterator based on verbose
    if verbose:
        iterator = tqdm(original_file.keys())
        pbar = iterator
    else:
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
    # ensure the path exists
    assert Path(hh).exists(), f"Path '{hh}' does not exist."
    # vh
    if not isinstance(vh, str):
        raise TypeError(f"Input value 'vh' type is {type(vh)}, but expected str.")
    assert Path(vh).exists(), f"Path '{vh}' does not exist."
    
    prefix = hh.split("data/")[0]

    hh = Path(hh)
    vh = Path(vh)

    interim_gm = Path(prefix + 'data/interim/gm')
    interim_gm.mkdir(parents=True, exist_ok=True)


    for file in [hh, vh]:
        with h5py.File(file, 'r') as f:
            with h5py.File(interim_gm / f"converted_{file.name}", 'w') as converted_file:
                convert_autopi_can(f, converted_file, verbose=True)