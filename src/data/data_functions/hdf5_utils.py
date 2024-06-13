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
import pandas as pd
from typing import Optional
from pathlib import Path

# ========================================================================================================================
#           hdf5 utility functions
# ========================================================================================================================

def unpack_hdf5(hdf5_file: str) -> dict:
    """
    Wrapper function used to call the recursive unpack function for unpacking the hdf5 file

    Parameters
    ----------
    hdf5_file : str
        The path to the hdf5 file

    Returns
    -------
    dict
        The unpacked data
    """
    if not isinstance(hdf5_file, str):
        raise TypeError(f"Input value 'hdf5_file' type is {type(hdf5_file)}, but expected str.")
    if not Path(hdf5_file).exists():
        raise FileNotFoundError(f"File '{hdf5_file}' does not exist.")
    if not Path(hdf5_file).suffix == '.hdf5':
        raise ValueError(f"File '{hdf5_file}' is not a hdf5 file.")
    
    with h5py.File(hdf5_file, 'r') as f:
        data = unpack_hdf5_(f)
    return data

def unpack_hdf5_(group: h5py.Group) -> dict:
    """
    Recursive function that unpacks the hdf5 file into a dictionary

    Parameters
    ----------
    group : h5py.Group
        The hdf5 group to unpack
    
    Returns
    -------
    dict
        The unpacked data
    """
    if not isinstance(group, h5py.Group | h5py.File):
        raise TypeError(f"Input value 'group' type is {type(group)}, but expected h5py.Group or h5py.File.")

    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            data[key] = unpack_hdf5_(group[key])
            data[key] = unpack_hdf5_(group[key])
        else:
            d = group[key][()]
            if isinstance(d, bytes):
                data[key] = d.decode('utf-8')
            else:
                data[key] = group[key][()]
    return data

def save_hdf5(data: dict, hdf5_file: str, segment_id: Optional[str] = None, attributes: Optional[dict] = None) -> None:
    """
    Wrapper function used to call the recursive save function for saving the data to an hdf5 file.

    Parameters
    ----------
    data : dict
        The data to save
    hdf5_file : str
        The path to the hdf5 file that we want to save the data to
    segment_id : str
        The segment id to save the data to. If None, the data is saved to the root of the hdf5 file
    """
    if not Path(hdf5_file).suffix == '.hdf5':
        raise ValueError(f"File '{hdf5_file}' is not a hdf5 file.")

    if segment_id is None:
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs.update(attributes)
            save_hdf5_(data, f)
    else:
        with h5py.File(hdf5_file, 'a') as f:
            segment_group = f.create_group(str(segment_id))
            segment_group.attrs.update(attributes)
            save_hdf5_(data, segment_group)

def save_hdf5_(data: dict, group: h5py.Group) -> None:
    """
    Recursive save function that saves the data to an hdf5 file

    Parameters
    ----------
    data : dict
        The data to save
    group : h5py.Group
        The hdf5 group to save the data to
    """
    if not isinstance(data, dict):
        raise TypeError(f"Input value 'data' type is {type(data)}, but expected dict.")
    if not isinstance(group, h5py.Group | h5py.File):
        raise TypeError(f"Input value 'group' type is {type(group)}, but expected h5py.Group or h5py.File.")
    
    for key, value in data.items():
        key = key.replace('/', '_')
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_hdf5_(value, subgroup)
        else:
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, pd.Series) and not value.dtype == 'O':
                group.create_dataset(key, data=value.values)
            elif isinstance(value, str):
                group.create_dataset(key, data=value.encode('utf-8'))