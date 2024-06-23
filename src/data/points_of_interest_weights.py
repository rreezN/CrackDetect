import numpy as np
import h5py
from typing import Any, Dict, List
import os



def load_from_hdf5(filename: str) -> Dict[str, Any]:
    """Load data from an HDF5 file and convert it to a nested dictionary structure.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to be loaded.

    Returns
    -------
    Dict[str, Any]
        A nested dictionary containing the data from the HDF5 file, 
        where groups are represented as dictionaries and datasets are converted to lists.

    """
    def unpack_group(group):
        unpacked_data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                unpacked_data[key] = unpack_group(item)
            else:
                unpacked_data[key] = item[:].tolist()  # Convert dataset to list
        return unpacked_data

    with h5py.File(filename, 'r') as hdf_file:
        loaded_mapping = unpack_group(hdf_file)
    
    return loaded_mapping


def read_from_hdf5(filename: str) -> Dict[int, List[List[Any]]]:
    """Read and convert data from an HDF5 file to a dictionary with specific conversions.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to be read.

    Returns
    -------
    Dict[int, List[List[Any]]]
        A dictionary where keys are integers and values are lists of trips. 
        Each trip is a list of items with specific type conversions applied.

    """
    with h5py.File(filename, 'r') as hdf_file:
        data = {}
        for index in hdf_file.keys():
            group = hdf_file[index]
            trips = []
            for trip_name in group.keys():
                trip = group[trip_name][:]
                # Convert back to appropriate types if needed
                converted_trip = []
                for pos, item in enumerate(trip): # TODO the car name should be a string 
                    try:
                        if pos <= 1:
                            item = item.decode('utf-8')
                            converted_trip.append(item)
                        elif pos > 1 and pos <= 3:
                            item = item.decode('utf-8')
                            converted_trip.append(int(float(item)))
                        elif pos > 3:
                            converted_trip.append(float(item))
                    except ValueError:
                        if isinstance(item, bytes):
                            # Handle byte strings
                            item = item.decode('utf-8')
                        converted_trip.append(item)
                trips.append(converted_trip)
            data[int(index)] = trips
    return data


def ln_of_ratio_sum_to_1(values: np.ndarray) -> np.ndarray:
    """Calculate normalized logarithm of ratio sums to 1 for a given array of values.

    Parameters
    ----------
    values : np.ndarray
        An array of numerical values.

    Returns
    -------
    np.ndarray
        An array of normalized logarithmic ratios.
    """
    total_values = np.sum(values)
    weight_values = -np.log(values / total_values)
    total = np.sum(weight_values)
    weight_norms = weight_values / total
    return weight_norms


def calculate_weights(mapping: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]) -> Dict[int, List[List[Any]]]:
    """Calculate weights for indexes based on a given mapping of data.

    Parameters
    ----------
    mapping : Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
        A nested dictionary containing data to calculate weights from.

    Returns
    -------
    Dict[int, List[List[Any]]]
        A dictionary where keys are indexes and values are lists of weights 
        for cars, trips, segments, seconds, and their corresponding weight ratios.
    """
    weights_for_indexes = {}
    indexes = np.sort([int(x) for x in mapping])
    for index in indexes:
        cars = []
        passes = []
        segments = []
        seconds = []
        weights_for_indexes[index] = []
        distances = []

        # Now go into each individual trip and segment and calculate the weight
        for car in mapping[str(index)]:
            for trip in mapping[str(index)][car]:
                for distance_segment_second in mapping[str(index)][car][trip]:
                    cars.append(car)
                    passes.append(trip)
                    distances.append(mapping[str(index)][car][trip][distance_segment_second][0])
                    segments.append(mapping[str(index)][car][trip][distance_segment_second][1])
                    seconds.append(mapping[str(index)][car][trip][distance_segment_second][2])

        if len(distances) == 1:
            weight_ln_ratio = [1.0]
        else:
            weight_ln_ratio = ln_of_ratio_sum_to_1(distances)

        for i in range(len(distances)):
            weights_for_indexes[index].append([cars[i], passes[i], segments[i], seconds[i], weight_ln_ratio[i]])

    return weights_for_indexes


def save_to_hdf5(mapping: Dict[int, List[List[Any]]], direction: str) -> None:
    """Save a mapping of data to an HDF5 file with a specified direction.

    Parameters
    ----------
    mapping : Dict[int, List[List[Any]]]
        A dictionary where keys are indexes and values are lists of trips.
        Each trip is a list of items to be saved.
    direction : str
        The direction to be used in the filename.

    Returns
    -------
    None
        This function does not return any value.

    """
    name = f"AOI_weighted_mapping_{direction}.hdf5"
    filename = f"data/AOI/{name}"

    if not os.path.exists('data/AOI'):
        os.makedirs('data/AOI')

    with h5py.File(filename, 'w') as hdf_file:
        # Create the HDF5 groups and datasets
        for index, trips in mapping.items():
            index_group = hdf_file.create_group(str(index))
            for trip in trips:
                # Convert each element to string and then to numpy array
                trip_array = np.array([str(item) for item in trip], dtype=h5py.string_dtype())
                index_group.create_dataset(str(trip_array), data=trip_array)

    print("HDF5 file has been created successfully.")


def main():
    # Takes about 40 seconds to load
    mapping_with_weights_hh = load_from_hdf5("data/AOI/mapping_hh_time_to_location.hdf5")
    mapping_with_weights_vh = load_from_hdf5("data/AOI/mapping_vh_time_to_location.hdf5")
    print("Loaded mapping")

    # Get weights for each index
    weights_for_indexes_hh = calculate_weights(mapping_with_weights_hh)
    weights_for_indexes_vh = calculate_weights(mapping_with_weights_vh)

    # Save the weights to a file
    save_to_hdf5(weights_for_indexes_hh, "hh")
    save_to_hdf5(weights_for_indexes_vh, "vh")
    
    # Try to load the saved file
    # loaded_weights_for_indexes_hh = read_from_hdf5("data/AOI/AOI_weighted_mapping_hh.hdf5")
    # loaded_weights_for_indexes_vh = read_from_hdf5("data/AOI/AOI_weighted_mapping_hh.hdf5")

if __name__ == "__main__":
    main()

