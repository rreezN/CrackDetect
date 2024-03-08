
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

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

def unpack_hdf5(hdf5_file, convert: bool = False):
    with h5py.File(hdf5_file, 'r') as f:
        data = unpack_hdf5_(f, convert)
    return data

def unpack_hdf5_(group, convert: bool = False):
    data = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            data[key] = unpack_hdf5_(group[key])
        else:
            if convert and key in parameter_dict:
                data[key] = convertdata(group[key][()], parameter_dict[key])
            else:
                d = group[key][()]
                if isinstance(d, bytes):
                    data[key] = d.decode('utf-8')
                else:
                    data[key] = group[key][()]
    return data

def save_hdf5(data, hdf5_file):
    with h5py.File(hdf5_file, 'w') as f:
        save_hdf5_(data, f)

def save_hdf5_(data, group):
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_hdf5_(value, subgroup)
        else:
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, str):
                group.create_dataset(key, data=value.encode('utf-8'))
    

def segment_gm(autopi: dict, direction: str, speed_threshold: int = 5, time_threshold: int = 10):
    # direction is either 'hh' or 'vh'
    pbar = tqdm(autopi.items())
    segment_index = 0
    for trip_name, trip in pbar:
        for pass_name, pass_ in trip.items():
            pbar.set_description(f"Interpolating {trip_name} {pass_name}")
            segments = segment_gm_trip(pass_, trip_name, pass_name, direction, speed_threshold=speed_threshold, time_threshold=time_threshold)
            for segment in segments:
                # Save the segment dictionary to a hdf5 file
                segment_path = Path(f'data/interim/gm/segment_{segment_index:03d}.hdf5')
                save_hdf5(segment, segment_path)

                # Increment the segment index
                segment_index += 1

def segment_gm_trip(measurements: dict, trip_name: str, pass_name: str, direction: str, speed_threshold: int = 5, time_threshold: int = 10):
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

def find_best_start_and_end_indeces(trip: np.ndarray, section: np.ndarray):
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index, end_index

def cut_dataframe_by_indeces(df, start, end):
    return df.iloc[start:end]

def interpolate(x, y, x_new):
    # Interpolate y values for x_new using x and y
    return np.interp(x_new, x, y)

def remove_duplicates(time, value):
    # Remove duplicate timestamps
    time_mask = np.concatenate((np.array([True]), np.diff(time) > 0))
    return time[time_mask], value[time_mask]

def resample_gm(section: dict, frequency: int = 250):
    # Resample the gm data to a fixed frequency by interpolating the measurements by distance

    # Calculate the distance between each point
    time, speed = remove_duplicates(
        section['measurements']['spd_veh'][:, 0],
        section['measurements']['spd_veh'][:, 1]
    )
    distance = np.cumsum(speed[:-1] * (time[1:] - time[:-1]) / 3.6)
    distance = np.cumsum(speed[:-1] * (time[1:] - time[:-1]) / 3.6)
    distance = np.insert(distance, 0, 0)

    start_time = time[0]
    end_time = time[-1]
    resampled_time = np.arange(start_time, end_time, 1/frequency)
    n_samples = len(resampled_time)
    resampled_distance = interpolate(time, distance, resampled_time)

    # Create a new section pd dataframe
    new_section = {
        "trip_name": np.repeat(section["trip_name"], n_samples),
        "pass_name": np.repeat(section["pass_name"], n_samples),
        "direction": np.repeat(section["direction"], n_samples),
    }

    new_section["time"] = resampled_time
    new_section["distance"] = resampled_distance

    for key, measurement in section['measurements'].items():
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

    # Convert the new section to a pd dataframe
    new_section = pd.DataFrame(new_section)
    return new_section

def resample_p79(section: pd.DataFrame, resampled_distances: np.ndarray):
    distance = section["Distance [m]"].values - section["Distance [m]"].values.min()
    new_section = {
        "distance": resampled_distances,
    }
    for key in section.columns:
        new_section[key] = interpolate(distance, section[key].values, resampled_distances)
    new_section = pd.DataFrame(new_section)
    return new_section

def resample_aran(section: pd.DataFrame, resampled_distances: np.ndarray):
    distance = np.abs(section["BeginChainage"].values - section["BeginChainage"].values[0])
    new_section = {
        "distance": resampled_distances,
    }
    for key in section.columns:
        if section[key].values.dtype == 'O':
            # Skip object columns
            continue
        new_section[key] = interpolate(distance, section[key].fillna(0).values, resampled_distances)
    new_section = pd.DataFrame(new_section)
    return new_section

def convert():
    # TODO: Rotate coordinates 
    # Save gm data with converted values
    autopi_hh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5', convert=True)
    autopi_vh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5', convert=True)

    Path('data/raw/gm').mkdir(parents=True, exist_ok=True)
    save_hdf5(autopi_hh, 'data/raw/gm/converted_platoon_CPH1_HH.hdf5')
    save_hdf5(autopi_vh, 'data/raw/gm/converted_platoon_CPH1_VH.hdf5')

def segment():
    # Load data
    autopi_hh = unpack_hdf5('data/raw/gm/converted_platoon_CPH1_HH.hdf5', convert=False)
    autopi_vh = unpack_hdf5('data/raw/gm/converted_platoon_CPH1_VH.hdf5', convert=False)

    # Create folders for saving
    Path('data/interim/gm').mkdir(parents=True, exist_ok=True)

    # Segment data
    segment_gm(autopi_hh['GM'], 'hh')
    segment_gm(autopi_vh['GM'], 'vh')

def match_data():
    # Find gm segment files
    segment_files = glob('data/interim/gm/*.hdf5')

    # Load reference and GoPro data
    aran_hh = pd.read_csv('data/raw/ref_data/cph1_aran_hh.csv', sep=';', encoding='unicode_escape')
    aran_vh = pd.read_csv('data/raw/ref_data/cph1_aran_vh.csv', sep=';', encoding='unicode_escape')

    p79_hh = pd.read_csv('data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape')
    p79_vh = pd.read_csv('data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')

    # Create folders for saving
    Path('data/interim/aran').mkdir(parents=True, exist_ok=True)
    Path('data/interim/p79').mkdir(parents=True, exist_ok=True)

    # Match data
    pbar = tqdm(segment_files)
    for i, segment_file in enumerate(pbar):
        pbar.set_description(f"Matching {segment_file.split('/')[-1]}")
        segment = unpack_hdf5(segment_file)

        segment_lonlat = segment['measurements']['gps'][:, 2:0:-1]

        if segment['direction'] == 'hh':
            # Match to ARAN data
            aran_hh_match = find_best_start_and_end_indeces(aran_hh[["Lon", "Lat"]].values, segment_lonlat)
            aran_segment = cut_dataframe_by_indeces(aran_hh, *aran_hh_match)
            aran_segment.to_csv(f'data/interim/aran/segment_{i:03d}.csv', sep=';', index=False)

            # Match to P79 data
            p79_hh_match = find_best_start_and_end_indeces(p79_hh[["Lon", "Lat"]].values, segment_lonlat)
            p79_segment = cut_dataframe_by_indeces(p79_hh, *p79_hh_match)
            p79_segment.to_csv(f'data/interim/p79/segment_{i:03d}.csv', sep=';', index=False)
        else:
            # Match to ARAN data
            aran_vh_match = find_best_start_and_end_indeces(aran_vh[["Lon", "Lat"]].values, segment_lonlat)
            aran_segment = cut_dataframe_by_indeces(aran_vh, *aran_vh_match)
            aran_segment.to_csv(f'data/interim/aran/segment_{i:03d}.csv', sep=';', index=False)

            # Match to P79 data
            p79_vh_match = find_best_start_and_end_indeces(p79_vh[["Lon", "Lat"]].values, segment_lonlat)
            p79_segment = cut_dataframe_by_indeces(p79_vh, *p79_vh_match)
            p79_segment.to_csv(f'data/interim/p79/segment_{i:03d}.csv', sep=';', index=False)

def resample():
    # Resample the gm data to a fixed frequency
    segment_files = glob('data/interim/gm/*.hdf5')

    Path('data/processed/gm').mkdir(parents=True, exist_ok=True)
    Path('data/processed/aran').mkdir(parents=True, exist_ok=True)
    Path('data/processed/p79').mkdir(parents=True, exist_ok=True)

    pbar = tqdm(segment_files)
    for i, segment_file in enumerate(pbar):
        pbar.set_description(f"Resampling {segment_file.split('/')[-1]}")
        segment = unpack_hdf5(segment_file)
        resampled_segment = resample_gm(segment)
        resampled_segment.to_csv(f'data/processed/gm/segment_{i:03d}.csv', sep=';', index=False)

        # Resample the P79
        p79_segment = pd.read_csv(f'data/interim/p79/segment_{i:03d}.csv', sep=';')
        resampled_distances = resampled_segment["distance"].values
        resampled_p79_segment = resample_p79(p79_segment, resampled_distances)
        resampled_p79_segment.to_csv(f'data/processed/p79/segment_{i:03d}.csv', sep=';', index=False)
        
        # Resample the ARAN
        aran_segment = pd.read_csv(f'data/interim/aran/segment_{i:03d}.csv', sep=';').fillna(0)
        resampled_aran_segment = resample_aran(aran_segment, resampled_distances)
        resampled_aran_segment.to_csv(f'data/processed/aran/segment_{i:03d}.csv', sep=';', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, default='segment', choices=['convert', 'segment', 'match', 'resample', 'all'], help='Mode to run the script in (all runs all modes in sequence)')

    args = parser.parse_args()

    if args.mode in ['convert', 'all']:
        print('    ---### Converting data ###---')
        convert()

    if args.mode in ['segment', 'all']:
        print('    ---### Segmenting data ###---')
        segment()

    if args.mode in ['match', 'all']:
        print('    ---###  Matching data  ###---')
        match_data()
    
    if args.mode in ['resample', 'all']:
        print('    ---### Resampling data ###---')
        resample()
