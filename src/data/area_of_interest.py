import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import re
import csv


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
        print(f)
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


def find_best_start_and_end_indeces_by_lonlat(trip: np.ndarray, section: np.ndarray):
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index, end_index


def natural_key(string):
    """A key to sort strings that contain numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


def define_trips_and_passes(segments):
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


def get_locations(p79, gm_data):
    # Extract every 10th item starting from idx[0] to idx[1]+1, to get one location for each meter
    lon_zp = p79['Lon']
    lat_zp = p79['Lat']
    idx = find_best_start_and_end_indeces_by_lonlat(p79[['Lat', 'Lon']].to_numpy(), gm_data['gps'][:,1:]) # TODO is this really the best place to start?? 
    loc_lon = lon_zp[idx[0]:idx[1]+1:10]
    loc_lat = lat_zp[idx[0]:idx[1]+1:10]

    # Combine lon and lat into a list of lists
    locations = [[lon, lat] for lon, lat in zip(loc_lon, loc_lat)]
    
    return locations
    

def map_time_to_area_of_interst(segments, locations, all_trip_names, pass_lists, direction):
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
            
             
            for second in segments[str(segment)].keys():
                current_second = segments[str(segment)][str(second)]
                current_second_lat = current_second["gm"][:,15]
                current_second_lon = current_second["gm"][:,16]
                current_second_locations = [[lon, lat] for lon, lat in zip(current_second_lon, current_second_lat)]
                closest_sample_arg = np.argmin(np.linalg.norm(np.column_stack((current_second_lon, current_second_lat)) - np.array([real_location[0], real_location[1]]), axis=1))
                best_at_second = current_second_locations[closest_sample_arg]
                distance = np.linalg.norm(np.array(best_at_second) - np.array(real_location))
                
                # Threshold for distance is set to 1.e-04, such that we only consider locations that are close enough
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


def filter_entries(data):
    # Here we filter out the entries that are not close enough to the real location or still have placeholder values
    threshold = 1.e-4
    
    # Function to filter the values within each pass
    def filter_pass(pass_data):
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


def save_to_csv(mapping, direction):
    name = f"mapping_{direction}_time_to_location.csv"
    filename = f"data/AOI/{name}"
    
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


def save_to_hdf5(mapping, direction):
    name = f"mapping_{direction}_time_to_location.hdf5"
    filename = f"data/AOI/{name}"

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
    segments = h5py.File('data/processed/segments.hdf5', 'r')
    all_trip_names, pass_lists = define_trips_and_passes(segments)
    
    # Right side data
    autopi_hh = unpack_hdf5('data/raw/AutoPi_CAN/platoon_CPH1_HH.hdf5')
    gm_data_hh = autopi_hh['GM']['16006']['pass_1'] # pass_1 is a VH route
    p79_hh = pd.read_csv('data/raw/ref_data/cph1_zp_hh.csv', sep=';', encoding='unicode_escape')
    
    # Left side data
    autopi_vh = unpack_hdf5(f'data/raw/AutoPi_CAN/platoon_CPH1_VH.hdf5')
    gm_data_vh = autopi_vh['GM']['16006']['pass_2'] # pass_2 is a VH route
    p79_vh = pd.read_csv('data/raw/ref_data/cph1_zp_vh.csv', sep=';', encoding='unicode_escape')   
    
    # Go trough both sides (takes 2 hours to run on Macbook 2019 16GB RAM 2.6 GHz 6-Core Intel Core i7)
    gm_data = [gm_data_hh, gm_data_vh] # The specific gm_data is only used to find_best_start_and_end_indeces_by_lonlat
    p79 = [p79_hh, p79_vh]
    directions = ['hh', 'vh']
    for gm_data_, p79_, direction in zip(gm_data, p79, directions):
        locations = get_locations(p79_, gm_data_)
        mapping = map_time_to_area_of_interst(segments, locations, all_trip_names, pass_lists, direction)
        cleaned_mapping = filter_entries(mapping) # TODO add funciton that cleans up the mapping dictionary right away
        save_to_csv(cleaned_mapping, direction)
        save_to_hdf5(cleaned_mapping, direction)


if __name__ == "__main__":
    main()