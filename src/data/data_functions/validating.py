import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import warnings
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm
from typing import Iterable



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

    Returns
    -------
    np.ndarray
        The accumulated distance in meters
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

    Returns
    -------
    np.ndarray
        The interpolated response values
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

def validate(hh: str = 'data/interim/gm/converted_platoon_CPH1_HH.hdf5', vh: str = 'data/interim/gm/converted_platoon_CPH1_VH.hdf5', threshold: float = 0.2, verbose: bool = False) -> None:
    """
    Validate the data by comparing the AutoPi data and the CAN data (car sensors)

    Parameters
    ----------
    threshold : float (default 0.2)
        The threshold for the normalised MSE between the AutoPi and CAN data
    verbose : bool (default False)
        Whether to plot the data for visual inspection
    """

    for file in [hh, vh]:
        with h5py.File(file, 'r+') as f:
            for trip_name, trip in (pbar := tqdm(f['GM'].items())):
                for pass_name, pass_ in trip.items():
                    pbar.set_description(f"File: {os.path.basename(file)}, Validating {trip_name}, {pass_name}")
                    validate_pass(pass_, threshold, verbose)
        



def plot_sensors(ax, time: np.ndarray, sensors: Iterable[np.ndarray], labels: Iterable[str], \
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

    for sensor, label, style in zip(sensors, labels, styles):
        ax.plot(time, sensor, style, label=label)

    ax.legend()
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

def normalised_mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the normalised mean squared error between two signals.

    Parameters
    ----------
    x : np.ndarray
        The first signal
    y : np.ndarray
        The second signal

    Returns
    -------
    float
        The normalised mean squared error
    """
    return np.mean((x-y)**2) / np.mean(np.maximum(x**2, y**2))

def validate_pass(car: h5py.Group, threshold: float, verbose: bool = False) -> None:
    """
    Main validation function for validating the data by comparing the AutoPi data and the CAN data (car sensors).
    NOTE: This function is a translation of the MATLAB code from PLATOON_SENSOR_VAL.m by Asmus Skar.

    Parameters
    ----------
    car : h5py.Group
        The car data to validate
    threshold : float
        The threshold for the normalised MSE between the AutoPi and CAN data
    verbose : bool
        Whether to plot the data for visual inspection
    """

    # Create custom warn message (Used to tell the user that the sensors are reoriented without interrupting tqdm progress bar)
    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return '\n' + str(msg) + '\n'

    warnings.formatwarning = custom_formatwarning
    
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
    axrpi_100hz = clean_int(taccrpi-time_start_max, xaccrpi, time) * 9.81
    ayrpi_100hz = clean_int(taccrpi-time_start_max, yaccrpi, time) * 9.81
    aycan_100hz = clean_int(tatra-time_start_max, atra, time)
    axcan_100hz = clean_int(talon-time_start_max, alon, time)
    dis_100hz   = clean_int(tspd-time_start_max, dspd, time)
    lon_100hz   = clean_int(tgps-time_start_max, lon, time)
    lat_100hz   = clean_int(tgps-time_start_max, lat, time)
    odo_100hz   = clean_int(todo-time_start_max, odo, time)

    # ACCELERATION MEASURE
    #   x-acceleration
    x_err = normalised_mse(axrpi_100hz, axcan_100hz)
    if x_err > threshold:
        warnings.warn(f"Warning: Normalised MSE between Autopi and CAN x-acceleration sensors is above threshold: {x_err}")

    #   y-acceleration
    y_err = normalised_mse(ayrpi_100hz, aycan_100hz)
    if y_err > threshold:
        print(f"Warning: Normalised MSE between Autopi and CAN y-acceleration sensors is above threshold: {y_err}")
    
    # DISTANCE MEASURE
    # Define distance as by the odometer and GPS
    odo_dist = odo_100hz - odo_100hz[0]
    gps_dist = distance_gps(np.column_stack((lat_100hz, lon_100hz)))

    # Compute correlation between the two distance measures
    mse_gps = normalised_mse(dis_100hz, gps_dist)
    mse_odo = normalised_mse(dis_100hz, odo_dist)
    if mse_gps > threshold or mse_odo > threshold:
        print(f"Warning: Normalised MSE between Autopi and CAN odo sensors or GPS distance is above threshold: {mse_odo}, {mse_gps}")

    if verbose:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_sensors(axes[0], time, sensors=[axrpi_100hz, axcan_100hz], labels=['Autopi x-acceleration', 'CAN x-acceleration'], styles=['r-', 'b-'], ylabel='Acceleration [$m/s^2$]', xlabel='Time [s]', title=f"Normalised MSE: {x_err:.3f}")
        plot_sensors(axes[1], time, sensors=[ayrpi_100hz, aycan_100hz], labels=['Autopi y-acceleration', 'CAN y-acceleration'], styles=['r-', 'b-'], ylabel='Acceleration [$m/s^2$]', xlabel='Time [s]', title=f"Normalised MSE: {y_err:.3f}")
        plot_sensors(axes[2], time, sensors=[dis_100hz, gps_dist, odo_dist], labels=['Autopi distance', 'GPS distance', 'Odometer distance'], styles=['r-', 'b-', 'g-'], ylabel='Distance [m]', xlabel='Time [s]', title=f"Normalised MSE Autopi vs. (odo, gps): {mse_odo:.2f}, {mse_gps:.2f}")
        plt.tight_layout()
        plt.show()

