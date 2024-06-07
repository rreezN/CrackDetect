import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import h5py
from scipy.interpolate import CubicSpline
from copy import deepcopy
from tqdm import tqdm
import re
import csv
from pathlib import Path


def load_from_hdf5(filename):

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


def get_total_distance(data):
    first_values = []

    for trip_name, passes in data.items():
        for pass_name, segments in passes.items():
            for segment_name, values in segments.items():
                if segment_name in ['distance_segment_second_1', 'distance_segment_second_2']:
                    # Extract the first value and add to the list
                    first_values.append(values[0])

    total_distance = np.sum(first_values)
    return total_distance


# Weight functions 
def ln_of_ratio(values, total):
    weight_values = -np.log(values / total)
    norm = np.linalg.norm(weight_values)
    unit_vector = weight_values / norm
    return unit_vector


def ln_of_ratio_sum_to_1(values, total):
    weight_values = -np.log(values / total)
    total = np.sum(weight_values)
    weight_values_1 = weight_values / total
    return weight_values_1


def inverse_of_ratio(values, total):
    weight_values = (values / total)**(-1)
    norm = np.linalg.norm(weight_values)
    unit_vector = weight_values / norm
    return unit_vector


def inverse_of_ratio_sum_to_1(values, total):
    weight_values = (values / total)**(-1)
    total = np.sum(weight_values)
    weight_values_1 = weight_values / total
    return weight_values_1


def calculate_weights(maps):
    weights_for_indexes = {}

    for mapping in maps:
        indexes = np.sort([int(x) for x in mapping])
        for index in indexes:
            weights_for_indexes[index] = []
            distances = []
            segments = []
            seconds = []

            total_distance_for_index = get_total_distance(mapping[str(index)])
            # Now go into each individual trip and segment and calculate the weight
            for car in mapping[str(index)]:
                for trip in mapping[str(index)][car]:
                    for distance_segment_second in mapping[str(index)][car][trip]:
                        distances.append(mapping[str(index)][car][trip][distance_segment_second][0])
                        segments.append(mapping[str(index)][car][trip][distance_segment_second][1])
                        seconds.append(mapping[str(index)][car][trip][distance_segment_second][2])

            
            weight_ln_ratio = ln_of_ratio_sum_to_1(distances, total_distance_for_index)
            weight_inverse_ratio = inverse_of_ratio_sum_to_1(distances, total_distance_for_index)

            for i in range(len(distances)):
                weights_for_indexes[index].append([distances[i], segments[i], seconds[i], weight_ln_ratio[i], weight_inverse_ratio[i]])

    return weights_for_indexes



def main():
    # Takes about 40 seconds to load
    mapping_with_weights_hh = load_from_hdf5("/data/AOI/mapping_hh_time_to_location.hdf5")
    mapping_with_weights_vh = load_from_hdf5("/data/AOI/mapping_hh_time_to_location.hdf5")

    maps = [mapping_with_weights_hh, mapping_with_weights_vh]

    weights_for_indexes = calculate_weights(maps)

    debug = 1

if __name__ == "__main__":
    main()

