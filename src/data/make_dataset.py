
import numpy as np
import pandas as pd
import h5py
import datetime as dt
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


def save_hdf5(data, hdf5_file, segment_id: str = None):
    if segment_id is None:
        with h5py.File(hdf5_file, 'w') as f:
            save_hdf5_(data, f)
    else:
        with h5py.File(hdf5_file, 'a') as f:
            segment_group = f.create_group(str(segment_id))
            save_hdf5_(data, segment_group)


def save_hdf5_(data, group):
    for key, value in data.items():
        key = key.replace('/', '_')
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_hdf5_(value, subgroup)
        else:
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, pd.Series):
                group.create_dataset(key, data=value.values)
            elif isinstance(value, str):
                group.create_dataset(key, data=value.encode('utf-8'))
    

def segment_gm(autopi: dict, direction: str, speed_threshold: int = 5, time_threshold: int = 10, segment_index: int = 0):
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


def find_best_start_and_end_indeces_by_lonlat(trip: np.ndarray, section: np.ndarray):
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index, end_index


def find_best_start_and_end_indeces_by_time(current_segment, gopro_time):
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


def cut_dataframe_by_indeces(df, start, end):
    return df.iloc[start:end]


def interpolate(x, y, x_new):
    # Interpolate y values for x_new using x and y
    return np.interp(x_new, x, y)


def remove_duplicates(time, value):
    # Remove duplicate timestamps
    time_mask = np.concatenate((np.array([True]), np.diff(time) > 0))
    return time[time_mask], value[time_mask]


def calculate_distance_from_time_and_speed(time, speed, conversion_factor=1):
    # Calculate the distance from the time and speed measurements
    distance = np.cumsum(speed[:-1] * (time[1:] - time[:-1]) / conversion_factor)
    distance = np.insert(distance, 0, 0)
    return distance


def resample_gm(section: h5py.Group, frequency: int = 250):
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
        "trip_name": section["trip_name"],
        "pass_name": section["pass_name"],
        "direction": section["direction"],
        "measurements": {}
    }

    new_section["measurements"]["time"] = resampled_time
    new_section["measurements"]["distance"] = resampled_distance

    for key, measurement in section['measurements'].items():
        measurement = measurement[()]
        measurement_time, measurement_value = remove_duplicates(measurement[:, 0], measurement[:, 1:])

        # Interpolate distance by time
        measurement_distance = interpolate(time, distance, measurement_time)
        # Interpolate measurements by distance
        if measurement_value.shape[1] > 1:
            # If the measurement is not 1D, add a column for each dimension
            for i in range(measurement_value.shape[1]):
                new_section["measurements"][f"{key}_{i}"] = interpolate(measurement_distance, measurement_value[:, i], resampled_distance)
        else:
            new_section["measurements"][key] = interpolate(measurement_distance, measurement_value.flatten(), resampled_distance)

    return new_section


def resample_gopro(section: h5py.Group, resampled_distances: np.ndarray):
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


# def resample_p79(section: h5py.Group, resampled_distances: np.ndarray):
#     distance_array = section["Distance [m]"][()]
#     distance = distance_array - distance_array.min()
#     new_section = {
#         "distance": resampled_distances,
#     }
#     for key, measurement in section.items():
#         new_section[key] = interpolate(distance, measurement[()], resampled_distances)
#     return new_section


# def resample_aran(section: pd.DataFrame, resampled_distances: np.ndarray):
#     distance = np.abs(section["BeginChainage"].values - section["BeginChainage"].values[0])
#     new_section = {
#         "distance": resampled_distances,
#     }
#     for key in section.columns:
#         if section[key].values.dtype == 'O':
#             # Skip object columns
#             continue
#         new_section[key] = interpolate(distance, section[key].fillna(0).values, resampled_distances)
#     new_section = pd.DataFrame(new_section)
#     return new_section


def csv_files_together(car_trip, go_pro_names, car_number):
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
            
        # save gopro_data[measurement
        new_folder = f"data/interim/gopro/{car_trip}"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        
        gopro_data.to_csv(f"{new_folder}/{measurement}.csv", index=False)


def convert():
    # TODO: Rotate coordinates 
    # Save gm data with converted values
    autopi_hh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5', convert=True)
    autopi_vh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5', convert=True)

    Path('data/interim/gm').mkdir(parents=True, exist_ok=True)
    save_hdf5(autopi_hh, 'data/interim/gm/converted_platoon_CPH1_HH.hdf5')
    save_hdf5(autopi_vh, 'data/interim/gm/converted_platoon_CPH1_VH.hdf5')
    
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
        pbar.set_description(f"Converting {car_trip}")
        csv_files_together(car_trip, car_gopro[car_trip], car_numbers[car_trip])
        

def segment():
    # Load data
    autopi_hh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_HH.hdf5', convert=False)
    autopi_vh = unpack_hdf5('data/interim/gm/converted_platoon_CPH1_VH.hdf5', convert=False)

    # Remove old segment file if it exists
    segment_path = Path('data/interim/gm/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()

    # Segment data
    segment_index = segment_gm(autopi_hh['GM'], 'hh')
    segment_gm(autopi_vh['GM'], 'vh', segment_index=segment_index)


def match_data():
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
                
            # gopro is a little different..
            if trip_name not in ["16006", "16009", "16011"]:
                continue
            
            gopro_segment = {}
            for measurement in ['gps5', 'accl', 'gyro']:
                start_index, end_index, start_diff, end_diff = find_best_start_and_end_indeces_by_time(segment, gopro_data[trip_name][measurement]["date"])

                if max(start_diff, end_diff) > 1:
                    continue
                
                # TODO when i=2 gopor_segment is empty..?
                gopro_segment[measurement] = gopro_data[trip_name][measurement][start_index:end_index].to_dict('series')

            save_hdf5(gopro_segment, 'data/interim/gopro/segments.hdf5', segment_id=i)

def resample():
# Resample the gm data to a fixed frequency
    gm_segment_file = 'data/interim/gm/segments.hdf5'
    
    Path('data/processed/gm').mkdir(parents=True, exist_ok=True)
    Path('data/processed/aran').mkdir(parents=True, exist_ok=True)
    Path('data/processed/p79').mkdir(parents=True, exist_ok=True)
    Path('data/processed/gopro').mkdir(parents=True, exist_ok=True)

    for folder in ["aran", "p79", "gopro", "gm"]:
        segment_path = Path(f'data/processed/{folder}/segments.hdf5')
        if segment_path.exists():
            segment_path.unlink()
        
    with h5py.File(gm_segment_file, 'r') as f:
        segment_files = [f[str(i)] for i in range(len(f))]
        pbar = tqdm(segment_files)
        with h5py.File('data/interim/gopro/segments.hdf5', 'r') as f2:
            with h5py.File('data/processed/gopro/segments.hdf5', 'a') as f3:
                for i, segment in enumerate(pbar):
                    pbar.set_description(f"Resampling segment {i+1:03d}/{len(segment_files)}")
                    segment_gopro_subgroup = f3.create_group(str(i))

                    # Resample the GM data
                    resampled_gm_segment = resample_gm(segment)
                    resampled_distances = resampled_gm_segment["measurements"]["distance"]
                    
                    # TODO Resample the GoPro data
                    if str(i) in f2.keys():       
                        gopro_segment = f2[str(i)]
                        resampled_gopro_segment = resample_gopro(gopro_segment, resampled_distances)
                        # save the resampled gopro data in groups of 250 in a hdf5 file
                        for j in range(250, len(resampled_gopro_segment["distance"]), 250):
                            time_subgroup = segment_gopro_subgroup.create_group(str(int(j/250)))
                            for key, value in resampled_gopro_segment.items():
                                    values = value[j-250:j]
                                    time_subgroup.create_dataset(key, data=values)
        
        debug = 1
                
                # Split GM, GoPro, P79 and ARAN data into 1 second segments


                # resampled_gopro_segment = resample_gopro(accl, gps5, gyro, resampled_distances)
                # resampled_gopro_segment.to_csv(f'data/processed/gopro/segment_{i:03d}.csv', sep=';', index=False)

                
                # Resample the P79
                # with h5py.File('data/interim/p79/segments.hdf5', 'r') as f:
                #     p79_segment = f[str(i)]
                #     resampled_p79_segment = resample_p79(p79_segment, resampled_distances)
                
                # Resample the ARAN
                # with h5py.File('data/interim/aran/segments.hdf5', 'r') as f:
                #     aran_segment = f[str(i)]
                #     resampled_aran_segment = resample_aran(aran_segment, resampled_distances)
        
        
            
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