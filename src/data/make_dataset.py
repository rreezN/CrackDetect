
import numpy as np
import pandas as pd
import h5py
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from typing import Optional, Iterable, List
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from scipy.interpolate import PchipInterpolator

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
    'acc.xyz':     {'kind': 'lowess', 'frac': 0.005},
    'spd_veh':     {'kind': 'lowess', 'frac': 0.005},
    'acc_long':     {'kind': 'lowess', 'frac': 0.005},
    'acc_trans':     {'kind': 'lowess', 'frac': 0.005}
}

"""
TODO:
- Better type hints
- Add paper references / docstrings
- Improve validation
"""


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
    """
    with h5py.File(hdf5_file, 'r') as f:
        data = unpack_hdf5_(f)
    return data

def unpack_hdf5_(group: h5py.Group) -> dict:
    """
    Recursive function that unpacks the hdf5 file into a dictionary
    """
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

def save_hdf5(data: dict, hdf5_file: str, segment_id: str = None) -> None:
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
    if segment_id is None:
        with h5py.File(hdf5_file, 'w') as f:
            save_hdf5_(data, f)
    else:
        with h5py.File(hdf5_file, 'a') as f:
            segment_group = f.create_group(str(segment_id))
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
    """
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
    """

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

    This function assumes the following:
        - The data is stored in hdf5 format
        - The data is stored, from the root at "data/raw/AutoPi_CAN/", which then allows 
            for the converted data to be stored in "data/interim/gm"

    Parameters
    ----------
    hh : str
        The path to the AutoPi CAN data for the HH direction
    vh : str
        The path to the AutoPi CAN data for the VH direction
    """
    hh = Path(hh)
    vh = Path(vh)

    interim_gm = Path('data/interim/gm')
    interim_gm.mkdir(parents=True, exist_ok=True)


    for file in [hh, vh]:
        with h5py.File(file, 'r') as f:
            with h5py.File(interim_gm / f"converted_{file.name}", 'w') as converted_file:
                convert_autopi_can(f, converted_file, verbose=True)


# ========================================================================================================================
#           Hardcoded GoPro functions
# ========================================================================================================================

def csv_files_together(car_trip: str, go_pro_names: list[str], car_number: str) -> None:
    """
    Saves the GoPro data to a csv file for each trip

    Parameters
    ----------
    car_trip : str
        The trip name
    go_pro_names : list[str]
        The names of the GoPro cameras
    car_number : str
        The car number
    """
    # Load all the gopro data 
    for measurement in ['accl', 'gps5', 'gyro']:
        gopro_data = None
        for trip_id in go_pro_names:
            trip_folder = f"data/raw/gopro_data/{car_number}/{trip_id}"
            new_data = pd.read_csv(f'{trip_folder}/{trip_id}_HERO8 Black-{measurement.upper()}.csv')
            new_data['date'] = pd.to_datetime(new_data['date']).map(dt.datetime.timestamp)
            
            # Drop all non-float columns
            new_data = new_data.select_dtypes(include=['float64', 'float32'])
        
            if gopro_data is not None:
                gopro_data = pd.concat([gopro_data, new_data])
            else:
                gopro_data = new_data
            
        # save gopro_data[measurement]
        new_folder = f"data/interim/gopro/{car_trip}"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        
        gopro_data.to_csv(f"{new_folder}/{measurement}.csv", index=False)

def preprocess_gopro_data() -> None:
    """
    Preprocess the GoPro data by combining the data from the three GoPro cameras into one csv file for each trip

    NOTE: This function is hardcoded for the three trips in the CPH1 dataset
    """

    # Create gopro data for the three trips
    car_trips = ["16011", "16009", "16006"]
    car_gopro = {
        "16011": ["GH012200", "GH022200", "GH032200", "GH042200", "GH052200", "GH062200"],
        "16009": ["GH010053", "GH030053", "GH040053", "GH050053", "GH060053"],
        "16006": ["GH020056", "GH040053"]
    }
    car_numbers = {
        "16011": "car1",
        "16009": "car3",
        "16006": "car3"
    }
    
    pbar = tqdm(car_trips)
    for car_trip in pbar:
        pbar.set_description(f"Converting GoPro/{car_trip}")
        csv_files_together(car_trip, car_gopro[car_trip], car_numbers[car_trip])


# ========================================================================================================================
#           Validation functions
# ========================================================================================================================

def distance_gps(gps: np.ndarray) -> np.ndarray:
    """
    Based on gps coordinates (lat, lon) calculate the distance in meters between each point, and returns the accumulated distance in meters.
    NOTE: This function is a translation of the MATLAB code from DISTANCE_GPS.m by Asmus Skar.

    Parameters
    ----------
    gps : np.ndarray
        The gps coordinates (lat, lon) in degrees
    """
    # Extract lat and lon
    lat = gps[:, 0]
    lon = gps[:, 1]

    # Create an array for the accumulated distance
    dx = np.zeros(len(lat))

    R = 6378.137 * 1e3  # Radius of Earth in m

    # Loop through the gps coordinates and calculate the distance
    for i in range(len(dx)-1):
        dLat = np.radians(lat[i+1] - lat[i])
        dLon = np.radians(lon[i+1] - lon[i])
        a = np.sin(dLat/2)**2 + np.cos(np.radians(lat[i])) * np.cos(np.radians(lat[i+1])) * np.sin(dLon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dx[i+1] = dx[i] + R * c

    return dx

def clean_int(tick: np.ndarray, response: np.ndarray, tick_int: np.ndarray) -> np.ndarray:
    """
    Using PCHIP interpolation, interpolate to the new tick_int values.
    NOTE: This function is a translation of the MATLAB code from CLEAN_INT.m by Asmus Skar.

    Parameters
    ----------
    tick : np.ndarray
        The tick values
    response : np.ndarray
        The response values
    tick_int : np.ndarray
        The tick values to interpolate to

    """
    # Add offset to multiple data (in interpolant)
    ve = np.cumsum(np.ones_like(tick)) * np.abs(tick) * np.finfo(float).eps  # Scaled Offset For Non-Zero Elements
    ve += np.cumsum(np.ones_like(tick)) * (tick == 0) * np.finfo(float).eps  # Add Scaled Offset For Zero Elements
    vi = tick + ve  # Interpolation Vector
    tick2 = vi

    # Create a PCHIP interpolator and interpolate
    pchip_interpolator = PchipInterpolator(tick2, response)
    data_int = pchip_interpolator(tick_int)

    return data_int

def validate(threshold: float, verbose: bool = False) -> None:
    """
    Validate the data by comparing the AutoPi data and the CAN data (car sensors)

    Parameters
    ----------
    threshold : float (default 0.9)
        The threshold for the correlation between the AutoPi and CAN data
    verbose : bool (default False)
        Whether to plot the data for visual inspection
    """
    # Save gm data with converted values
    autopi_hh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_HH.hdf5')
    autopi_vh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_VH.hdf5')

    iterator = tqdm([autopi_hh, autopi_vh])

    # Validate the trips
    for file in iterator:
        for trip_name, trip in file['GM'].items():
            for pass_name, pass_ in trip.items():
                iterator.set_description(f"Validating {trip_name}/{pass_name}")
                validate_pass(pass_, threshold, verbose)

def plot_sensors(time: np.ndarray, sensors: Iterable[np.ndarray], labels: Iterable[str], \
                 styles: Iterable[str], ylabel: str = None, xlabel: str = None, title: str = None) -> None:
    """
    Plot function to visualise the sensor data and their correlation when validating the data and verbose is set to True.

    Parameters
    ----------
    time : np.ndarray
        The time values
    sensors : Iterable[np.ndarray]
        The sensor data to plot
    labels : Iterable[str]
        The labels for the sensors
    styles : Iterable[str]
        The styles for the sensors
    ylabel : str
        The y-axis label
    xlabel : str
        The x-axis label
    title : str
        The title of the plot
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for sensor, label, style in zip(sensors, labels, styles):
        ax.plot(time, sensor, style, label=label)

    ax.legend()
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

def validate_pass(car: dict, threshold: float, verbose: bool = False) -> None:
    """
    Main validation function for validating the data by comparing the AutoPi data and the CAN data (car sensors).
    NOTE: This function is a translation of the MATLAB code from PLATOON_SENSOR_VAL.m by Asmus Skar.

    Parameters
    ----------
    car : dict
        The car data to validate
    threshold : float
        The threshold for the correlation between the AutoPi and CAN data
    verbose : bool
        Whether to plot the data for visual inspection
    """
    
    fs = 10 # Sampling frequency

    # Speed distance
    tspd = car['spd_veh'][:, 0]
    dspd = np.cumsum(car['spd_veh'][1:, 1]*np.diff(car['spd_veh'][:, 0]))/3.6
    dspd = np.insert(dspd, 0, 0)

    # GPS data
    tgps = car['gps'][:, 0]
    lat  = car['gps'][:, 1]
    lon  = car['gps'][:, 2]

    # Odometer
    todo = car['odo'][:, 0]
    odo = car['odo'][:, 1] * 1e3 + np.cumsum(car['f_dist'][:, 1]) * 1e-2

    # Normalize accelerations
    taccrpi = car['acc.xyz'][:, 0]
    xaccrpi = car['acc.xyz'][:, 1] - np.mean(car['acc.xyz'][:, 1])
    yaccrpi = car['acc.xyz'][:, 2] - np.mean(car['acc.xyz'][:, 2])
    zaccrpi = car['acc.xyz'][:, 3] - np.mean(car['acc.xyz'][:, 3])

    tatra = car['acc_trans'][:, 0]
    atra = car['acc_trans'][:, 1] - np.mean(car['acc_trans'][:, 1])
    talon = car['acc_long'][:, 0]
    alon = car['acc_long'][:, 1] - np.mean(car['acc_long'][:, 1])

    # Resample to 100Hz
    time_start_max = np.max([taccrpi[0], tatra[0], talon[0], tgps[0], tspd[0], todo[0]])
    time_end_min = np.min([taccrpi[-1], tatra[-1], talon[-1], tgps[-1], tspd[-1], todo[-1]])
    tend = time_end_min - time_start_max
    time = np.arange(0, tend, 1/fs)

    # Interpolate
    axrpi_100hz = clean_int(taccrpi-time_start_max, xaccrpi, time)
    ayrpi_100hz = clean_int(taccrpi-time_start_max, yaccrpi, time)
    azrpi_100hz = clean_int(taccrpi-time_start_max, zaccrpi, time)
    aycan_100hz = clean_int(tatra-time_start_max, atra, time)
    axcan_100hz = clean_int(talon-time_start_max, alon, time)
    dis_100hz   = clean_int(tspd-time_start_max, dspd, time)
    lon_100hz   = clean_int(tgps-time_start_max, lon, time)
    lat_100hz   = clean_int(tgps-time_start_max, lat, time)
    odo_100hz   = clean_int(todo-time_start_max, odo, time)

    # Reorient accelerations
    alon = axcan_100hz.copy()
    atrans = aycan_100hz.copy()
    axpn = axrpi_100hz * 9.81
    aypn = ayrpi_100hz * 9.81
    azpn = azrpi_100hz * 9.81

    # Calculate correlation with CAN accelerations
    pcxl = np.corrcoef(axpn, alon)[0, 1]
    pcyl = np.corrcoef(aypn, alon)[0, 1]

    pcxt = np.corrcoef(axpn, atrans)[0, 1]
    pcyt = np.corrcoef(aypn, atrans)[0, 1]
    
    # Determine the orientation of the sensors
    if (abs(pcxl) < abs(pcxt)) and (abs(pcyl) > abs(pcyt)):
        if pcxt < 0:
            axrpi_100hz = aypn
            ayrpi_100hz = -axpn
        else:
            axrpi_100hz = aypn
            ayrpi_100hz = axpn
    else:
        if pcyt < 0:
            axrpi_100hz = axpn
            ayrpi_100hz = -aypn
        else:
            axrpi_100hz = axpn
            ayrpi_100hz = aypn
    
    # Update the acceleration sensors with the reoriented values
    axcan_100hz = alon
    aycan_100hz = atrans
    azrpi_100hz = azpn

    # ACCELERATION MEASURE
    #   x-acceleration
    pcx = np.corrcoef(axrpi_100hz, axcan_100hz)[0, 1]
    if pcx < threshold:
        print(f"Correlation between Autopi and CAN x-acceleration sensors is below threshold: {pcx}")
        if verbose:
            plot_sensors(time, sensors=[axrpi_100hz, axcan_100hz], labels=['Autopi x-acceleration', 'CAN x-acceleration'], styles=['r-', 'b-'], ylabel='Acceleration [$m/s^2$]', xlabel='Time [s]', title=f"Correlation: {pcx:.3f}")

    #   y-acceleration
    pcy = np.corrcoef(ayrpi_100hz, aycan_100hz)[0, 1]

    if pcy < threshold:
        print(f"Correlation between Autopi and CAN y-acceleration sensors is below threshold: {pcy}")
        if verbose:
            plot_sensors(time, sensors=[ayrpi_100hz, aycan_100hz], labels=['Autopi y-acceleration', 'CAN y-acceleration'], styles=['r-', 'b-'], ylabel='Acceleration [$m/s^2$]', xlabel='Time [s]', title=f"Correlation: {pcy:.3f}")
    
    # #   z-acceleration
    # pcz = np.corrcoef(azrpi_100hz, np.zeros_like(azrpi_100hz))[0, 1]

    # DISTANCE MEASURE
    # Define distance as by the odometer and GPS
    odo_dist = odo_100hz - odo_100hz[0]
    gps_dist = distance_gps(np.column_stack((lat_100hz, lon_100hz)))

    # Compute correlation between the two distance measures
    pcgps = np.corrcoef(dis_100hz, gps_dist)[0, 1]
    pcodo = np.corrcoef(dis_100hz, odo_dist)[0, 1]
    if pcodo < threshold or pcgps < threshold:
        print(f"Correlation between Autopi and CAN odo sensors or GPS distance is below threshold: {pcodo}, {pcgps}")
        if verbose:
            plot_sensors(time, sensors=[dis_100hz, gps_dist, odo_dist], labels=['Autopi distance', 'GPS distance', 'Odometer distance'], styles=['r-', 'b-', 'g-'], ylabel='Distance [m]', xlabel='Time [s]', title=f"Correlation Autopi vs. (odo, gps): {pcodo:.3f}, {pcgps:.3f}")
        
# ========================================================================================================================
#           Segmentation functions
# ========================================================================================================================

def segment_gm(autopi: dict, direction: str, speed_threshold: int = 5, time_threshold: int = 10, segment_index: int = 0) -> int:
    """
    Segment the GM data into sections where the vehicle is moving
    
    Parameters
    ----------
    autopi : dict
        The AutoPi data dictionary
    direction : str
        The direction of the trip, either 'hh' or 'vh'
    speed_threshold : int
        The speed in km/h below which the vehicle is considered to be stopped
    time_threshold : int
        The minimum time in seconds for a section to be considered valid
    segment_index : int
        The index to start the segment numbering from
    """
    # direction is either 'hh' or 'vh'
    pbar = tqdm(autopi.items())
    for trip_name, trip in pbar:
        for pass_name, pass_ in trip.items():
            pbar.set_description(f"Interpolating {trip_name} {pass_name}")
            segments = segment_gm_trip(pass_, trip_name, pass_name, direction, speed_threshold=speed_threshold, time_threshold=time_threshold)
            for segment in segments:
                # Save the segment dictionary to a hdf5 file
                segment_path = Path(f'data/interim/gm/segments.hdf5')
                save_hdf5(segment, segment_path, segment_id=segment_index)

                # Increment the segment index
                segment_index += 1
    return segment_index


def segment_gm_trip(measurements: dict, trip_name: str, pass_name: str, direction: str, speed_threshold: int = 5, time_threshold: int = 10):
    """
    Segment a single GM trip into sections where the vehicle is moving
    
    Parameters
    ----------
    measurements : dict
        The AutoPi data dictionary
    trip_name : str
        The name of the trip
    pass_name : str
        The name of the pass
    direction : str
        The direction of the trip, either 'hh' or 'vh'
    speed_threshold : int
        The speed in km/h below which the vehicle is considered to be stopped
    time_threshold : int
        The minimum time in seconds for a section to be considered valid
    """

    # threshold is the speed in km/h below which the vehicle is considered to be stopped
    measurements["spd_veh"][:, 1] = measurements["spd_veh"][:, 1]

    # Find the ranges in time where the speed is not zero
    non_zero_speed_ranges = []
    start_index = -1
    for i in range(len(measurements["spd_veh"])):
        if measurements["spd_veh"][i, 1] > speed_threshold:
            if start_index == -1:
                start_index = i
        else:
            if start_index != -1:
                non_zero_speed_ranges.append((measurements["spd_veh"][start_index, 0], measurements["spd_veh"][i, 0]))
                start_index = -1
    if start_index != -1:
        non_zero_speed_ranges.append((measurements["spd_veh"][start_index, 0], measurements["spd_veh"][-1, 0]))

    # Create a list of dictionaries, each containing the measurements for a section of the trip
    sections = []
    for start, end in non_zero_speed_ranges:
        # Check if the section is too short
        if end - start < time_threshold:
            continue
        section = {
            "trip_name": trip_name,
            "pass_name": pass_name,
            "direction": direction,
            "measurements": {}
        }
        for key, value in measurements.items():
            section["measurements"][key] = value[(value[:, 0] >= start) & (value[:, 0] <= end)]
        sections.append(section)
    
    return sections

def segment(speed_threshold: int = 5, time_threshold: int = 10) -> None:
    """
    Segment the GM data into sections where the vehicle is moving.

    Parameters
    ----------
    speed_threshold : int
        The speed in km/h below which the vehicle is considered to be stopped
    time_threshold : int
        The minimum time in seconds for a section to be considered valid
    """
    
    # Load data
    autopi_hh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_HH.hdf5')
    autopi_vh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_VH.hdf5')

    # Remove old segment file if it exists
    segment_path = Path('data/interim/gm/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()

    # Segment data
    segment_index = segment_gm(autopi_hh['GM'], direction='hh', speed_threshold=speed_threshold, time_threshold=time_threshold)
    segment_gm(autopi_vh['GM'], direction='vh', speed_threshold=speed_threshold, time_threshold=time_threshold, segment_index=segment_index)


# ========================================================================================================================
#           Matching functions
# ========================================================================================================================

def match_data() -> None:
    # Define path to segment files
    segment_file = 'data/interim/gm/segments.hdf5'

    # Load reference and GoPro data
    aran = {
        'hh': pd.read_csv('data/raw/ref_data/cph1_aran_hh.csv', sep=';', encoding='unicode_escape').fillna(0),
        'vh': pd.read_csv('data/raw/ref_data/cph1_aran_vh.csv', sep=';', encoding='unicode_escape').fillna(0)
    }

    p79 = {
        'hh': pd.read_csv('data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape'),
        'vh': pd.read_csv('data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')
    }
    
    gopro_data = {}
    car_trips = ["16011", "16009", "16006"]
    for trip_id in car_trips:
        gopro_data[trip_id] = {}
        for measurement in ['gps5', 'accl', 'gyro']:
            gopro_data[trip_id][measurement] = pd.read_csv(f'data/interim/gopro/{trip_id}/{measurement}.csv')

    # Create folders for saving
    Path('data/interim/aran').mkdir(parents=True, exist_ok=True)
    Path('data/interim/p79').mkdir(parents=True, exist_ok=True)
    Path('data/interim/gopro').mkdir(parents=True, exist_ok=True)

    # Remove old segment files if they exist
    for folder in ['aran', 'p79', 'gopro']:
        segment_path = Path(f'data/interim/{folder}/segments.hdf5')
        if segment_path.exists():
            segment_path.unlink()

    # Match data
    with h5py.File(segment_file, 'r') as f:
        segment_files = [f[str(i)] for i in range(len(f))]
        pbar = tqdm(segment_files)
        for i, segment in enumerate(pbar):
            pbar.set_description(f"Matching segment {i+1:03d}/{len(segment_files)}")

            direction = segment['direction'][()].decode("utf-8")
            trip_name = segment["trip_name"][()].decode('utf-8')
            pass_name = segment["pass_name"][()].decode('utf-8')

            segment_lonlat = segment['measurements']['gps'][()][:, 2:0:-1]

            aran_dir = aran[direction]
            p79_dir = p79[direction]

            # Match to ARAN data
            aran_match = find_best_start_and_end_indeces_by_lonlat(aran_dir[["Lon", "Lat"]].values, segment_lonlat)
            aran_segment = cut_dataframe_by_indeces(aran_dir, *aran_match)
            save_hdf5(aran_segment, 'data/interim/aran/segments.hdf5', segment_id=i)

            # Match to P79 data
            p79_match = find_best_start_and_end_indeces_by_lonlat(p79_dir[["Lon", "Lat"]].values, segment_lonlat)
            p79_segment = cut_dataframe_by_indeces(p79_dir, *p79_match)
            save_hdf5(p79_segment, 'data/interim/p79/segments.hdf5', segment_id=i)
                
            # gopro is a little different.. (These trips do not have any corresponding gopro data, so we skip them)
            if trip_name not in ["16006", "16009", "16011"]:
                continue
            
            gopro_segment = {}
            for measurement in ['gps5', 'accl', 'gyro']:
                start_index, end_index, start_diff, end_diff = find_best_start_and_end_indeces_by_time(segment, gopro_data[trip_name][measurement]["date"])

                if max(start_diff, end_diff) > 1:
                    continue
                
                gopro_segment[measurement] = gopro_data[trip_name][measurement][start_index:end_index].to_dict('series')

            if gopro_segment != {}:
                save_hdf5(gopro_segment, 'data/interim/gopro/segments.hdf5', segment_id=i)


# ========================================================================================================================
#           Resampling functions
# ========================================================================================================================

def find_best_start_and_end_indeces_by_lonlat(trip: np.ndarray, section: np.ndarray) -> tuple[int, int]:
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index, end_index+1


def find_best_start_and_end_indeces_by_time(current_segment: h5py.Group, gopro_time: np.ndarray) -> tuple[int, int, float, float]:
    # Find the start and end indeces of the section data based on time
    
    current_segment_start_time = current_segment["measurements"]["gps"][()][0, 0]
    current_segment_end_time = current_segment["measurements"]["gps"][()][-1, 0]
    segment_time = [current_segment_start_time, current_segment_end_time]
    
    diff_start = (gopro_time - segment_time[0]).abs()
    start_index = diff_start.idxmin()
    start_diff = diff_start.min()
    
    diff_end = (gopro_time - segment_time[1]).abs()
    end_index = diff_end.idxmin()
    end_diff = diff_end.min()

    return start_index, end_index, start_diff, end_diff


def cut_dataframe_by_indeces(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df.iloc[start:end]


def interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    # Interpolate y values for x_new using x and y
    return np.interp(x_new, x, y)


def remove_duplicates(time: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove duplicate timestamps
    time_mask = np.concatenate((np.array([True]), np.diff(time) > 0))
    return time[time_mask], value[time_mask]


def calculate_distance_from_time_and_speed(time: np.ndarray, speed: np.ndarray, conversion_factor: int = 1) -> np.ndarray:
    # Calculate the distance from the time and speed measurements
    distance = np.cumsum(speed[:-1] * (time[1:] - time[:-1]) / conversion_factor)
    distance = np.insert(distance, 0, 0)
    return distance


def resample_gm(section: h5py.Group, frequency: int = 250) -> dict[str, np.ndarray]:
    # Resample the gm data to a fixed frequency by interpolating the measurements by distance

    # Calculate the distance between each point
    time, speed = remove_duplicates(
        section['measurements']['spd_veh'][()][:, 0],
        section['measurements']['spd_veh'][()][:, 1]
    )
    distance = calculate_distance_from_time_and_speed(time, speed, 3.6)

    start_time = time[0]
    end_time = time[-1]
    resampled_time = np.arange(start_time, end_time, 1/frequency)
    n_samples = len(resampled_time)
    resampled_distance = interpolate(time, distance, resampled_time)

    # Create a new section pd dataframe
    new_section = {
    }

    new_section["time"] = resampled_time
    new_section["distance"] = resampled_distance

    for key, measurement in section['measurements'].items():
        measurement = measurement[()]
        measurement_time, measurement_value = remove_duplicates(measurement[:, 0], measurement[:, 1:])

        # Interpolate distance by time
        measurement_distance = interpolate(time, distance, measurement_time)
        # Interpolate measurements by distance
        if measurement_value.shape[1] > 1:
            # If the measurement is not 1D, add a column for each dimension
            for i in range(measurement_value.shape[1]):
                new_section[f"{key}_{i}"] = interpolate(measurement_distance, measurement_value[:, i], resampled_distance)
        else:
            new_section[key] = interpolate(measurement_distance, measurement_value.flatten(), resampled_distance)

    return new_section


def resample_gopro(section: h5py.Group, resampled_distances: np.ndarray) -> dict[str, np.ndarray]:
    gps5 = section["gps5"]
    gps5_time, gps5_speed = gps5["date"][()], gps5["GPS (3D speed) [m_s]"][()]
    accl = section["accl"]
    accl_time = accl["date"][()]
    gyro = section["gyro"]
    gyro_time = gyro["date"][()]

    # Interpolate the speed measurements from the GPS data
    interpolate_accl_speed = interpolate(gps5_time, gps5_speed, accl_time)
    interpolate_gyro_speed = interpolate(gps5_time, gps5_speed, gyro_time)

    # Calculate distances
    measurement_distances = {
        "accl": calculate_distance_from_time_and_speed(accl_time, interpolate_accl_speed),
        "gps5": calculate_distance_from_time_and_speed(gps5_time, gps5_speed),
        "gyro": calculate_distance_from_time_and_speed(gyro_time, interpolate_gyro_speed)
    }

    new_section = {
        "distance": resampled_distances,
    }
    for name, measurement in zip(["accl", 'gps5', 'gyro'], [accl, gps5, gyro]):
        for key, value in measurement.items():
            if key in new_section.keys():
                # Skip object columns and duplicates
                continue
            new_section[key] = interpolate(measurement_distances[name], value[()], resampled_distances)
    return new_section

def resample(verbose: bool = False) -> None:
    # Resample the gm data to a fixed frequency
    frequency = 250
    seconds_per_step = 1

    aran_counts = []
    p79_counts = []

    gm_segment_file = 'data/interim/gm/segments.hdf5'
    
    Path('data/processed/wo_kpis').mkdir(parents=True, exist_ok=True)

    segment_path = Path(f'data/processed/wo_kpis/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()
    
    # Load raw reference data
    aran = {
        'hh': pd.read_csv('data/raw/ref_data/cph1_aran_hh.csv', sep=';', encoding='unicode_escape').fillna(0).select_dtypes(include=np.number),
        'vh': pd.read_csv('data/raw/ref_data/cph1_aran_vh.csv', sep=';', encoding='unicode_escape').fillna(0).select_dtypes(include=np.number)
    }

    p79 = {
        'hh': pd.read_csv('data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape').select_dtypes(include=np.number),
        'vh': pd.read_csv('data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape').select_dtypes(include=np.number)
    }

    # Resample the data
    with h5py.File(gm_segment_file, 'r') as f:
        segment_files = [f[str(i)] for i in range(len(f))]
        pbar = tqdm(segment_files)
        # Load gopro data
        with h5py.File('data/interim/gopro/segments.hdf5', 'r') as f2:
            # Open final processed segments file
            with h5py.File('data/processed/wo_kpis/segments.hdf5', 'a') as f3:
                for i, segment in enumerate(pbar):
                    pbar.set_description(f"Resampling segment {i+1:03d}/{len(segment_files)}")
                    segment_subgroup = f3.create_group(str(i))

                    # Add direction, trip name and pass name as attr to segment subgroup
                    segment_subgroup.attrs['direction'] = segment['direction'][()].decode("utf-8")
                    segment_subgroup.attrs['trip_name'] = segment["trip_name"][()].decode('utf-8')
                    segment_subgroup.attrs['pass_name'] = segment["pass_name"][()].decode('utf-8')

                    # Get relevant reference data
                    direction = segment['direction'][()].decode("utf-8")
                    aran_dir = aran[direction]
                    p79_dir = p79[direction]

                    # Resample the GM data
                    resampled_gm_segment = resample_gm(segment, frequency=frequency)
                    resampled_distances = resampled_gm_segment["distance"]

                    # Cut the aran and p79 data by the lonlat of the segment
                    bit_lon = resampled_gm_segment['gps_1']
                    bit_lat = resampled_gm_segment['gps_0']
                    bit_lonlat = np.column_stack((bit_lon, bit_lat))
                    aran_segment_match = find_best_start_and_end_indeces_by_lonlat(aran_dir[["Lon", "Lat"]].values, bit_lonlat)
                    aran_segment = cut_dataframe_by_indeces(aran_dir, *aran_segment_match)

                    p79_segment_match = find_best_start_and_end_indeces_by_lonlat(p79_dir[["Lon", "Lat"]].values, bit_lonlat)
                    p79_segment = cut_dataframe_by_indeces(p79_dir, *p79_segment_match)

                    # resample the gopro data
                    gopro_data_exists = False
                    if str(i) in f2.keys():
                        gopro_data_exists = True
                        gopro_segment = f2[str(i)]
                        resampled_gopro_segment = resample_gopro(gopro_segment, resampled_distances)

                    # Cut segments into 1 second bits
                    steps = (len(resampled_distances) // (frequency * seconds_per_step))
                    for j in range(steps):
                        start = j*frequency*seconds_per_step
                        end = (j+1)*frequency*seconds_per_step
                        time_subgroup = segment_subgroup.create_group(str(j))
                        
                        # concatenate the measurements for each 1 second bit
                        gm_measurements = []
                        gm_attributes = {}
                        for i, (measurement_key, measurement_value) in enumerate(resampled_gm_segment.items()):
                            gm_attributes[measurement_key] = i
                            gm_measurements.append(measurement_value[start: end])
                        gm_measurements = np.column_stack(gm_measurements)
                        # Save the resampled GM data in groups of 'frequency' length
                        gm_dataset = time_subgroup.create_dataset("gm", data=gm_measurements)
                        gm_dataset.attrs.update(gm_attributes)

                        if gopro_data_exists:
                            # save the resampled gopro data in groups of 'frequency' length
                            gopro_measurements = []
                            gopro_attributes = {}
                            for i, (key, value) in enumerate(resampled_gopro_segment.items()):
                                values = value[start: end]
                                gopro_attributes[key] = i
                                gopro_measurements.append(values)
                            gopro_measurements = np.column_stack(gopro_measurements)
                            gopro_dataset = time_subgroup.create_dataset("gopro", data=gopro_measurements)
                            gopro_dataset.attrs.update(gopro_attributes)
            
                        # Find the corresponding ARAN and P79 data for each 1 second bit using closest lonlat points
                        bit_lonlat_time = bit_lonlat[start: end]
                        aran_match_bit = find_best_start_and_end_indeces_by_lonlat(aran_segment[["Lon", "Lat"]].values, bit_lonlat_time)
                        aran_bit = cut_dataframe_by_indeces(aran_segment, *aran_match_bit).values
                        aran_columns = aran_segment.columns
                        aran_dataset = time_subgroup.create_dataset("aran", data=aran_bit)
                        for i, column in enumerate(aran_columns):
                            aran_dataset.attrs[column] = i
                        aran_counts.append(len(aran_bit))

                        p79_match_bit = find_best_start_and_end_indeces_by_lonlat(p79_segment[["Lon", "Lat"]].values, bit_lonlat_time)
                        p79_bit = cut_dataframe_by_indeces(p79_segment, *p79_match_bit).values
                        p79_columns = p79_segment.columns
                        p79_dataset = time_subgroup.create_dataset("p79", data=p79_bit)
                        for i, column in enumerate(p79_columns):
                            p79_dataset.attrs[column] = i
                        p79_counts.append(len(p79_bit))

                        # Plot the longitude and lattitude coordinates of the gm segment and the matched ARAN and P79 data
                        if verbose and (aran_match_bit[1] - aran_match_bit[0] < 3 or p79_match_bit[1] - p79_match_bit[0] < 3):
                            verbose_resample_plot(bit_lonlat_time, aran_segment, aran_match_bit, p79_segment, p79_match_bit)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].hist(aran_counts, bins=20)
    axes[0].set_title("ARAN segment length distribution")
    axes[0].set_xlabel("Number of points")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(p79_counts, bins=20)
    axes[1].set_title("P79 segment length distribution")
    axes[1].set_xlabel("Number of points")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def verbose_resample_plot(bit_lonlat_time, aran_segment, aran_match_bit, p79_segment, p79_match_bit):
    fig, ax = plt.subplots()
    ax.plot(bit_lonlat_time[:, 0], bit_lonlat_time[:, 1], label='GM', c='k')
    # Extract ARAN and P79 with n extra points on each side for better visualization
    n = 3
    aran_match_before = max(0, aran_match_bit[0] - n)
    aran_match_after = min(len(aran_segment), aran_match_bit[1] + n)
    wider_aran_bit = aran_segment[["Lon", "Lat"]].values[aran_match_before: aran_match_after]
    shallow_aran_bit = aran_segment[["Lon", "Lat"]].values[aran_match_bit[0]: aran_match_bit[1]]
    p79_match_before = max(0, p79_match_bit[0] - n)
    p79_match_after = min(len(p79_segment), p79_match_bit[1] + n)
    wider_p79_bit = p79_segment[["Lon", "Lat"]].values[p79_match_before: p79_match_after]
    shallow_p79_bit = p79_segment[["Lon", "Lat"]].values[p79_match_bit[0]: p79_match_bit[1]]
    ax.plot(wider_aran_bit[:, 0], wider_aran_bit[:, 1], label='ARAN', linestyle='--', alpha=0.5)
    ax.scatter(shallow_aran_bit[:, 0], shallow_aran_bit[:, 1], alpha=0.5, marker='x')
    if aran_match_before < aran_match_bit[0]:
        ax.scatter(aran_segment[["Lon", "Lat"]].values[aran_match_before: aran_match_bit[0], 0], aran_segment[["Lon", "Lat"]].values[aran_match_before: aran_match_bit[0], 1], alpha=0.5, marker='x', c='r')
    if aran_match_after > aran_match_bit[1]:
        ax.scatter(aran_segment[["Lon", "Lat"]].values[aran_match_bit[1]: aran_match_after, 0], aran_segment[["Lon", "Lat"]].values[aran_match_bit[1]: aran_match_after, 1], alpha=0.5, marker='x', c='r')
    ax.plot(wider_p79_bit[:, 0], wider_p79_bit[:, 1], label='P79', linestyle='--', alpha=0.5)
    ax.scatter(shallow_p79_bit[:, 0], shallow_p79_bit[:, 1], alpha=0.5, marker='x')
    if p79_match_before < p79_match_bit[0]:
        ax.scatter(p79_segment[["Lon", "Lat"]].values[p79_match_before: p79_match_bit[0], 0], p79_segment[["Lon", "Lat"]].values[p79_match_before: p79_match_bit[0], 1], alpha=0.5, marker='x', c='r')
    if p79_match_after > p79_match_bit[1]:
        ax.scatter(p79_segment[["Lon", "Lat"]].values[p79_match_bit[1]: p79_match_after, 0], p79_segment[["Lon", "Lat"]].values[p79_match_bit[1]: p79_match_after, 1], alpha=0.5, marker='x', c='r')
    # Place legend outside of plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


# ========================================================================================================================
#           KPI functions
# ========================================================================================================================



def compute_kpis(WINDOW_SIZES: list[int] = [1, 2]) -> None:
    """
    Alters the existing segments file by adding KPIs to each second in each segment, based on the window sizes provided.

    Do account for the fact that the first and last seconds of each segment depend on the max(WINDOW_SIZES), such that the KPIs can be computed,
    and compared across window-sizes.

    Parameters
    ----------
    WINDOW_SIZES : list[int]
        The window sizes to compute KPIs for. Default is [1, 2]. 
    """

    # Create folders for saving
    Path('data/processed/w_kpis').mkdir(parents=True, exist_ok=True)

    # Remove old segment files if they exist
    segment_path = Path(f'data/processed/w_kpis/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()
    
    # Load processed data
    with h5py.File('data/processed/wo_kpis/segments.hdf5', 'r') as f:
        # Open final processed segments file
        with h5py.File('data/processed/w_kpis/segments.hdf5', 'a') as f2:
            for i, segment in (pbar := tqdm(f.items())):
                pbar.set_description(f"Computing KPIs for segment {i}")
                segment_subgroup = f2.create_group(str(i))

                # Add direction, trip name and pass name as attr to segment subgroup
                segment_subgroup.attrs['direction'] = segment.attrs['direction']
                segment_subgroup.attrs['trip_name'] = segment.attrs['trip_name']
                segment_subgroup.attrs['pass_name'] = segment.attrs['pass_name']

                num_seconds_in_segment = len(segment)

                for j, second in segment.items():
                    j = int(j)
                    # Skip the first and last seconds which can not be computed with a window size of max(WINDOW_SIZES)
                    if j < max(WINDOW_SIZES) or j >= num_seconds_in_segment - max(WINDOW_SIZES):
                        continue

                    second_subgroup = segment_subgroup.create_group(str(j))

                    for key, value in second.items():
                        second_subgroup.create_dataset(key, data=value[()])
                        second_subgroup[key].attrs.update(second[key].attrs)
                    
                    # Compute KPIs
                    kpi_subgroup = second_subgroup.create_group('kpis')
                    kpi_subgroup.attrs['window_sizes'] = WINDOW_SIZES
                    for window_size in WINDOW_SIZES:
                        kpis = compute_kpis_for_second(segment, j, window_size)
                        kpi_data = kpi_subgroup.create_dataset(str(window_size), data=kpis)
                        for i, kpi_name in enumerate(['DI', 'RUT', 'PI', 'IRI']):
                            kpi_data.attrs[kpi_name] = i


def compute_kpis_for_second(segment: h5py.Group, second_index: int, window_size: int) -> np.ndarray:
    """
    Compute KPIs for a given second in a segment, based on a window size.

    Parameters
    ----------
    segment : h5py.Group
        The segment to compute KPIs for.
    second_index : int
        The index of the second to compute KPIs for.
    window_size : int
        The window size to compute KPIs for.
    """
    # Extract ARAN data for all seconds within the window
    windowed_aran_data = []
    for i in range(second_index - window_size, second_index + window_size + 1):
        windowed_aran_data.append(segment[str(i)]['aran'][()])
    
    # Define aran attributes for KPI-functions
    aran_attrs = segment[str(second_index)]['aran'].attrs

    # Stack the ARAN data
    windowed_aran_data = np.vstack(windowed_aran_data)

    # Compute KPIs
    # damage index
    KPI_DI = damage_index(windowed_aran_data, aran_attrs)
    # rutting index
    KPI_RUT = rutting_mean(windowed_aran_data, aran_attrs)
    # patching index
    PI = patching_sum(windowed_aran_data, aran_attrs)
    # IRI
    IRI = iri_mean(windowed_aran_data, aran_attrs)
    
    return np.asarray([KPI_DI, KPI_RUT, PI, IRI])


def damage_index(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Calculates the damage index for a given window of ARAN data as specified in the ARAN manual. NOTE TODO

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """

    crackingsum = cracking_sum(windowed_aran_data, aran_attrs)
    alligatorsum = alligator_sum(windowed_aran_data, aran_attrs)
    potholessum = pothole_sum(windowed_aran_data, aran_attrs)
    DI = crackingsum + alligatorsum + potholessum
    return DI


def cracking_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Conventional/longitudinal and transverse cracks are reported as length.

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """
    LCS = windowed_aran_data[:, aran_attrs['Revner På Langs Små (m)']]
    LCM = windowed_aran_data[:, aran_attrs['Revner På Langs Middelstore (m)']]
    LCL = windowed_aran_data[:, aran_attrs['Revner På Langs Store (m)']]
    TCS = windowed_aran_data[:, aran_attrs['Transverse Low (m)']]
    TCM = windowed_aran_data[:, aran_attrs['Transverse Medium (m)']]
    TCL = windowed_aran_data[:, aran_attrs['Transverse High (m)']]
    return ((LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1)).mean()


def alligator_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    alligator cracks are computed as area of the pavement affected by the damage

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    
    """
    ACS = windowed_aran_data[:, aran_attrs['Krakeleringer Små (m²)']]
    ACM = windowed_aran_data[:, aran_attrs['Krakeleringer Middelstore (m²)']]
    ACL = windowed_aran_data[:, aran_attrs['Krakeleringer Store (m²)']]
    return ((3*ACS + 4*ACM + 5*ACL)**(0.3)).mean()


def pothole_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """


    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """
    PAS = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Low (mm)']]
    PAM = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Medium (mm)']]
    PAL = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth High (mm)']]
    PAD = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Delamination (mm)']]
    return ((5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1)).mean()


def rutting_mean(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager, rut: str ='straight-edge') -> float:
    """
    

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """

    # TODO: FIGURE OUT WHICH ONE TO USE
    if rut == 'straight-edge':
        RDL = windowed_aran_data[:, aran_attrs['LRUT Straight Edge (mm)']]
        RDR = windowed_aran_data[:, aran_attrs['RRUT Straight Edge (mm)']]
    elif rut == 'wire':
        RDL = windowed_aran_data[:, aran_attrs['LRUT Wire (mm)']]
        RDR = windowed_aran_data[:, aran_attrs['RRUT Wire (mm)']]
    return (((RDL +RDR)/2)**(0.5)).mean()


def iri_mean(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """
    IRL = windowed_aran_data[:, aran_attrs['Venstre IRI (m/km)']]
    IRR = windowed_aran_data[:, aran_attrs['Højre IRI (m/km)']]
    return (((IRL + IRR)/2)**(0.2)).mean()
    
def patching_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    """
    LCSe = windowed_aran_data[:, aran_attrs['Revner På Langs Sealed (m)']]
    TCSe = windowed_aran_data[:, aran_attrs['Transverse Sealed (m)']]
    return ((LCSe**2 + 2*TCSe)**(0.1)).mean()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, default='segment', choices=['convert', 'validate', 'segment', 'match', 'resample', 'kpi', 'all'], help='Mode to run the script in (all runs all modes in sequence)')
    parser.add_argument('--speed-threshold', type=int, default=5, help='Speed threshold for segmenting data')
    parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for segmenting data')
    parser.add_argument('--validation-threshold', type=float, default=0.8, help='Threshold for validating data')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    if args.mode in ['convert', 'all']:
        print('    ---### Converting data ###---')
        convert()

        # Convert GoPro data to align with the GM trips
        preprocess_gopro_data()
    
    if args.mode in ['validate', 'all']:
        print('    ---### Validating data ###---')
        validate(threshold=args.validation_threshold, verbose=args.verbose)

    if args.mode in ['segment', 'all']:
        print('    ---### Segmenting data ###---')
        segment(args.speed_threshold, args.time_threshold)

    if args.mode in ['match', 'all']:
        print('    ---###  Matching data  ###---')
        match_data()
    
    if args.mode in ['resample', 'all']:
        print('    ---### Resampling data ###---')
        resample(args.verbose)
    
    if args.mode in ['kpi', 'all']:
        print('    ---### Calculating KPIs ###---')
        compute_kpis()