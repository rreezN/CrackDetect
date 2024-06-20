import numpy as np
import h5py


def read_from_hdf5(filename):
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


def  



def main():
    # Takes about 40 seconds to load
    weights_for_indexes_hh = read_from_hdf5("data/AOI/AOI_weighted_mapping_hh.hdf5")
    weights_for_indexes_vh = read_from_hdf5("data/AOI/AOI_weighted_mapping_hh.hdf5")

    KPI = "data/processed/w_kpis/segments.hdf5"
    debug = 1

if __name__ == "__main__":
    main()