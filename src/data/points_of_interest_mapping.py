import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm
from pathlib import Path
import re
import csv
from typing import Tuple, Any, Dict, List
import numpy as np
import os

parameter_dict = {
        'acc_long':     {'bstar': 198,      'rstar': 1,     'b': 198,   'r': 0.05   },
        'acc_trans':    {'bstar': 32768,    'rstar': 1,     'b': 32768, 'r': 0.04   },
        'acc_yaw':      {'bstar': 2047,     'rstar': 1,     'b': 2047,  'r': 0.1    },
        'brk_trq_elec': {'bstar': 4096,     'rstar': -1,    'b': 4098,  'r': -1     },
        'whl_trq_est':  {'bstar': 12800,    'rstar': 0.5,   'b': 12700, 'r': 1      },
        'trac_cons':    {'bstar': 80,       'rstar': 1,     'b': 79,    'r': 1      },
        'trip_cons':    {'bstar': 0,        'rstar': 0.1,   'b': 0,     'r': 1      }
    }


def convertdata(data: np.ndarray, parameter: Dict[str, float]) -> np.ndarray:
    """Convert data using provided parameters.

    Parameters
    ----------
    data : np.ndarray
        A numpy array where the first column is time and the second column is the data to be converted.
    parameter : Dict[str, float]
        A dictionary containing conversion parameters with keys 'bstar', 'rstar', 'b', and 'r'.

    Returns
    -------
    np.ndarray
        The converted data as a numpy array with the same shape as the input.
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


def unpack_hdf5(hdf5_file: str, convert: bool = False) -> Dict[str, Any]:
    """Unpack data from an HDF5 file, optionally converting it using predefined parameters.

    Parameters
    ----------
    hdf5_file : str
        Path to the HDF5 file to be unpacked.
    convert : bool, optional
        If True, convert data using predefined parameters, by default False.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the unpacked data.
    """
    with h5py.File(hdf5_file, 'r') as f:
        print(f)
        data = unpack_hdf5_(f, convert)
    return data


def unpack_hdf5_(group: h5py.Group, convert: bool = False) -> Dict[str, Any]:
    """Recursively unpack data from an HDF5 group, optionally converting it using predefined parameters.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to unpack.
    convert : bool, optional
        If True, convert data using predefined parameters, by default False.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the unpacked data.
    """
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


def find_best_start_and_end_indeces_by_lonlat(trip: np.ndarray, section: np.ndarray) -> Tuple[int, int]:
    """Find the start and end indices of the section data that are closest to the trip data.

    Parameters
    ----------
    trip : np.ndarray
        A numpy array with shape (n, 2), where n is the number of points. Each row represents 
        a point with longitude and latitude (lon, lat) of the trip data.
    section : np.ndarray
        A numpy array with shape (m, 2), where m is the number of points. Each row represents 
        a point with longitude and latitude (lon, lat) of the section data.

    Returns
    -------
    Tuple[int, int]
        The start and end indices in the trip data that are closest to the first and last points 
        of the section data, respectively.
    """
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index, end_index


def natural_key(string):
    """A key to sort strings that contain numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


def define_trips_and_passes(segments: h5py.File) -> Tuple[List[str], Dict[str, List[str]]]:
    """Define all trips and passes from the segments data.

    Parameters
    ----------
    segments : h5py.File
        The HDF5 file containing the segments data.

    Returns
    -------
    Tuple[List[str], Dict[str, List[str]]]
        A tuple containing a list of all trip names and a dictionary with pass lists for each trip.
    """
    
    # Define all_trip_names and pass_lists
    all_trip_names = ['16006', '16008', '16009', '16010', '16011']
    # Initialize the dictionary with empty sets for each trip
    pass_names_for_each_trip = {trip: set() for trip in all_trip_names}

    # Populate the sets with pass names from the segments
    for segment in segments.values():
        trip_name = segment.attrs["trip_name"]
        pass_name = segment.attrs["pass_name"]
        pass_names_for_each_trip[trip_name].add(pass_name)

    # Convert sets to sorted lists using the natural key for sorting
    for trip in pass_names_for_each_trip:
        pass_names_for_each_trip[trip] = sorted(pass_names_for_each_trip[trip], key=natural_key)

    # Variable assignments for pass lists
    for trip, passes in pass_names_for_each_trip.items():
        globals()[f'pass_list_{trip}'] = passes

    # Sorted pass list
    pass_lists = {trip: sorted(passes, key=natural_key) for trip, passes in pass_names_for_each_trip.items()}
        
    return all_trip_names, pass_lists


def get_locations(p79: pd.DataFrame, gm_data: Dict[str, np.ndarray]) -> List[List[float]]:
    """Extract every 10th location from the trip data that corresponds to the section data.

    Parameters
    ----------
    p79 : pd.DataFrame
        A pandas DataFrame containing the trip data with 'Lat' and 'Lon' columns.
    gm_data : Dict[str, np.ndarray]
        A dictionary containing the section data. It should have a key 'gps' with the corresponding
        GPS data as a numpy array.

    Returns
    -------
    List[List[float]]
        A list of locations where each location is a list containing longitude and latitude [lon, lat].
    """
    lon_zp = p79['Lon']
    lat_zp = p79['Lat']
    idx = find_best_start_and_end_indeces_by_lonlat(p79[['Lat', 'Lon']].to_numpy(), gm_data['gps'][:,1:]) # TODO is this really the best place to start?? 
    loc_lon = lon_zp[idx[0]:idx[1]+1:10]
    loc_lat = lat_zp[idx[0]:idx[1]+1:10]

    # Combine lon and lat into a list of lists
    locations = [[lon, lat] for lon, lat in zip(loc_lon, loc_lat)]
    
    return locations
    
# Add type hints to the function signature

def map_time_to_area_of_interst(segments: h5py.File, locations: List[List[float]], all_trip_names: List[str], pass_lists: Dict
[str, List[str]], direction: str) -> Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """Map the time to the two best seconds for each pass in each trip for each location.
    
    Parameters
    ----------
    segments : h5py.File
        The HDF5 file containing the segments data.
    locations : List[List[float]]
        A list of locations where each location is a list containing longitude and latitude [lon, lat].
    all_trip_names : List[str]
        A list of all trip names.
    pass_lists : Dict[str, List[str]]   
        A dictionary with pass lists for each trip.
    direction : str
        The direction of the segments to consider.

    Returns
    -------
    Dict[int, Dict[str, Dict[str, Dict[str, List[float]]]]]
        A dictionary where keys are indexes and values are dictionaries with trip names as keys and dictionaries with 
        pass names as keys. The values are dictionaries with segment names as keys and lists of the two best seconds 
        for each pass in each trip for each location.
    """
    # Initialize the mapping dictionary
    mapping_to_the_two_best_seconds_for_each_pass_in_each_trip = {}

    for index in range(len(locations)):
        trip_data = {}
        for trip_name in all_trip_names:
            pass_data = {}
            for pass_name in pass_lists[trip_name]:
                pass_data[pass_name] = {
                    "distance_segment_second_1": [100, 0, 0],
                    "distance_segment_second_2": [200, 0, 0] # these values are just placeholders
                }
            trip_data[trip_name] = pass_data
        mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index] = trip_data
    
    # Now fill out the dictionary with the correct values
    for index, real_location in tqdm(enumerate(locations), total=len(locations), desc="Processing"):
        for segment in segments.keys():
            current_direction = segments[str(segment)].attrs["direction"]
            
            # Check that we are going in the right direction
            if current_direction != direction:
                continue
            
            current_trip_name = segments[str(segment)].attrs["trip_name"]
            current_pass_name = segments[str(segment)].attrs["pass_name"]
            current_best_trip_1 = mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index][current_trip_name][current_pass_name]['distance_segment_second_1']
            current_best_trip_2 = mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index][current_trip_name][current_pass_name]['distance_segment_second_2']
            
            # Convert list of seconds to integers
            int_list = [int(x) for x in segments[str(segment)].keys()]
            # sort the list, remove the two smallest and two largest numbers and Convert seconds back to strings
            relevant_seconds = [str(x) for x in sorted(int_list)[2:-2]]
            
            for second in relevant_seconds:                
                current_second = segments[str(segment)][str(second)]
                current_second_lat = current_second["gm"][:,15]
                current_second_lon = current_second["gm"][:,16]
                current_second_locations = [[lon, lat] for lon, lat in zip(current_second_lon, current_second_lat)]
                closest_sample_arg = np.argmin(np.linalg.norm(np.column_stack((current_second_lon, current_second_lat)) - np.array([real_location[0], real_location[1]]), axis=1))
                best_at_second = current_second_locations[closest_sample_arg] # NOTE we could use avearge of all instead
                distance = np.linalg.norm(np.array(best_at_second) - np.array(real_location))
                
                # Threshold for distance is set to 1.e-04 (11.1 meters), such that we only consider locations that are close enough
                threshold_distance = 1.e-04
                if distance < threshold_distance:

                    if distance < current_best_trip_1[0]:
                        current_best_trip_2 = current_best_trip_1
                        current_best_trip_1 = [distance, segment, second]
                        mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index][current_trip_name][current_pass_name]['distance_segment_second_1'] = current_best_trip_1
                        mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index][current_trip_name][current_pass_name]['distance_segment_second_2'] = current_best_trip_2
                        
                    elif distance < current_best_trip_2[0]:
                        current_best_trip_2 = [distance, segment, second]
                        mapping_to_the_two_best_seconds_for_each_pass_in_each_trip[index][current_trip_name][current_pass_name]['distance_segment_second_2'] = current_best_trip_2
    

    return mapping_to_the_two_best_seconds_for_each_pass_in_each_trip


def filter_entries(data: Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]) -> Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]:
    """Filter out entries that are not close enough to the real location or still have placeholder values.

    Parameters
    ----------
    data : Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]
        A nested dictionary containing the data to be filtered.

    Returns
    -------
    Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]
        A nested dictionary containing the filtered data.
    """
    threshold = 1.e-4
    
    def filter_pass(pass_data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Filter the values within each pass based on the distance threshold.

        Parameters
        ----------
        pass_data : Dict[str, List[Any]]
            A dictionary containing segment data and values for each pass.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary containing the filtered pass data.
        """
        filtered_pass_data = {}
        for segment, values in pass_data.items():
            distance = values[0]
            if distance < threshold:
                filtered_pass_data[segment] = values
        return filtered_pass_data
    
    # Iterate through the main dictionary to filter the entries
    filtered_data = {}
    for key, trips in data.items():
        filtered_trips = {}
        for trip, passes in trips.items():
            filtered_passes = {}
            for pass_name, pass_data in passes.items():
                filtered_pass_data = filter_pass(pass_data)
                if filtered_pass_data:
                    filtered_passes[pass_name] = filtered_pass_data
            if filtered_passes:
                filtered_trips[trip] = filtered_passes
        if filtered_trips:
            filtered_data[key] = filtered_trips
            
    return filtered_data


def save_mapping_csv(mapping: Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]], direction: str, path_to_aoi: str = "data/AOI") -> None:
    """Save a mapping of data to a CSV file with a specified direction.

    Parameters
    ----------
    mapping : Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]
    direction : str
        The direction of the segments to consider.
    path_to_aoi : str
        The path to the area of interest, by default "data/AOI".

    Returns
    -------
    None
    """
    Path(path_to_aoi).mkdir(parents=True, exist_ok=True)
    name = f"mapping_{direction}_time_to_location.csv"
    filename = os.path.join(path_to_aoi, name)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Index', 'Trip Name', 'Pass Name', 'Distance Segment', 'Value1', 'Value2', 'Value3'])
        # Write the data
        for index, trips in mapping.items():
            for trip_name, passes in trips.items():
                for pass_name, segments in passes.items():
                    for segment_name, values in segments.items():
                        # Prepare the row with all needed information
                        row = [index, trip_name, pass_name, segment_name] + values
                        writer.writerow(row)

    print("CSV file has been created successfully.")


# Add type hints and docstrings to the function signature
def save_mapping_hdf5(mapping: Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]], direction: str, path_to_aoi: str = "data/AOI") -> None:
    """Save a mapping of data to an HDF5 file with a specified direction.

    Parameters
    ----------
    mapping : Dict[int, Dict[str, Dict[str, Dict[str, List[Any]]]]]
        A dictionary where keys are indexes and values are dictionaries with trip names as keys and dictionaries with 
        pass names as keys. The values are dictionaries with segment names as keys and lists of the two best seconds 
        for each pass in each trip for each location.
    direction : str
        The direction to be used in the filename.
    path_to_aoi : str
        The path to the area of interest, by default "data/AOI".

    Returns
    -------
    None
    """
    Path(path_to_aoi).mkdir(parents=True, exist_ok=True)
    name = f"mapping_{direction}_time_to_location.hdf5"
    filename = os.path.join(path_to_aoi, name)

    with h5py.File(filename, 'w') as hdf_file:
        # Create the HDF5 groups and datasets
        for index, trips in mapping.items():
            index_group = hdf_file.create_group(str(index))
            for trip_name, passes in trips.items():
                trip_group = index_group.create_group(trip_name)
                for pass_name, segments in passes.items():
                    pass_group = trip_group.create_group(pass_name)
                    for segment_name, values in segments.items():
                        # Convert values to a numerical array if possible
                        try:
                            values_array = np.array(values, dtype=np.float64)
                        except ValueError:
                            # Handle the case where conversion fails
                            values_array = np.array(values, dtype=h5py.string_dtype())

                        pass_group.create_dataset(segment_name, data=values_array)

    print("HDF5 file has been created successfully.")


def main():
    # Get the segments 
    segments = h5py.File('data/processed/w_kpis/segments.hdf5', 'r')
    all_trip_names, pass_lists = define_trips_and_passes(segments)
    
    # Right side data
    autopi_hh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5')
    gm_data_hh = autopi_hh['GM']['16006']['pass_1'] # uneven passes are HH routes
    p79_hh = pd.read_csv('data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape')
    
    # Left side data
    autopi_vh = unpack_hdf5(f'data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5')
    gm_data_vh = autopi_vh['GM']['16006']['pass_2'] # even passes are VH routes
    p79_vh = pd.read_csv('data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')   
    
    # Go trough both sides (takes 1 hour 15 m to run on Macbook 2019 16GB RAM 2.6 GHz 6-Core Intel Core i7)
    gm_data = [gm_data_hh, gm_data_vh] # The specific gm_data is only used to find_best_start_and_end_indeces_by_lonlat
    p79 = [p79_hh, p79_vh]
    directions = ['hh', 'vh']
    for gm_data_, p79_, direction in zip(gm_data, p79, directions):
        locations = get_locations(p79_, gm_data_)
        mapping = map_time_to_area_of_interst(segments, locations, all_trip_names, pass_lists, direction)
        cleaned_mapping = filter_entries(mapping) # TODO add funciton that cleans up the mapping dictionary right away
        save_mapping_csv(mapping=cleaned_mapping, direction=direction)
        save_mapping_hdf5(mapping=cleaned_mapping, direction=direction)


if __name__ == "__main__":
    main()