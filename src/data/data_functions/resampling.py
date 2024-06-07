import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from .matching import find_best_start_and_end_indeces_by_lonlat



# ========================================================================================================================
#           Resampling functions
# ========================================================================================================================

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
    """
    Resample the gm data to a fixed frequency by interpolating the measurements by distance

    Parameters
    ----------
    section : h5py.Group
        The section data
    frequency : int
        The frequency to resample to
    
    Returns
    -------
    new_section : dict[str, np.ndarray]
        The resampled section data
    """

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
    """
    Resample the gopro data to a fixed frequency by interpolating the measurements by distance

    Parameters
    ----------
    section : h5py.Group
        The section data
    resampled_distances : np.ndarray
        The resampled distances

    Returns
    -------
    new_section : dict[str, np.ndarray]
        The resampled section data
    """

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

def extract_bit_data(segment: h5py.Group, start: int, end: int) -> tuple[np.ndarray, dict[str, int]]:
    """
    Extract a 1 second reference data bit (start to end) from the segment

    Parameters
    ----------
    segment : h5py.Group
        The segment data
    start : int
        The start index
    end : int
        The end index

    Returns
    -------
    bit_data : np.ndarray
        The 1 second bit data
    """
    bit_data = np.zeros((end - start, len(segment.keys())))
    bit_attributes = {}
    for i, (key, value) in enumerate(segment.items()):
        bit_data[:, i] = value[start: end]
        bit_attributes[key] = i
    return bit_data, bit_attributes

def resample(verbose: bool = False) -> None:
    """
    Resample the GM data to a fixed frequency and save the resampled data into a new file
    Additionally resample the GoPro data if it exists

    Parameters
    ----------
    verbose : bool
        Whether to plot the resampled 1 second bits for visual inspection
    """

    frequency = 250
    seconds_per_step = 1

    aran_counts = []
    p79_counts = []

    gm_segment_file = 'data/interim/gm/segments.hdf5'
    
    Path('data/processed/wo_kpis').mkdir(parents=True, exist_ok=True)

    segment_path = Path(f'data/processed/wo_kpis/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()

    # Resample the data
    with h5py.File(gm_segment_file, 'r') as gm:
        segment_files = [gm[str(i)] for i in range(len(gm))]
        pbar = tqdm(segment_files)
        # Load gopro data
        with h5py.File('data/interim/gopro/segments.hdf5', 'r') as gopro:
            # Load ARAN and P79 data
            with h5py.File('data/interim/aran/segments.hdf5', 'r') as aran:
                with h5py.File('data/interim/p79/segments.hdf5', 'r') as p79:
                    # Open final processed segments file
                    with h5py.File('data/processed/wo_kpis/segments.hdf5', 'a') as wo_kpis:
                        for i, segment in enumerate(pbar):
                            pbar.set_description(f"Resampling segment {i+1:03d}/{len(segment_files)}")
                            segment_subgroup = wo_kpis.create_group(str(i))

                            # Add direction, trip name and pass name as attr to segment subgroup
                            segment_subgroup.attrs['direction'] = segment['direction'][()].decode("utf-8")
                            segment_subgroup.attrs['trip_name'] = segment["trip_name"][()].decode('utf-8')
                            segment_subgroup.attrs['pass_name'] = segment["pass_name"][()].decode('utf-8')

                            # Get relevant reference data
                            aran_segment = aran[str(i)]
                            aran_segment_lonlat = np.column_stack((aran_segment['Lon'], aran_segment['Lat']))
                            p79_segment = p79[str(i)]
                            p79_segment_lonlat = np.column_stack((p79_segment['Lon'], p79_segment['Lat']))

                            # Resample the GM data
                            resampled_gm_segment = resample_gm(segment, frequency=frequency)
                            resampled_distances = resampled_gm_segment["distance"]
                            gm_segment_lonlat = np.column_stack((resampled_gm_segment['gps_1'], resampled_gm_segment['gps_0']))

                            # resample the gopro data
                            gopro_data_exists = False
                            if str(i) in gopro.keys():
                                gopro_data_exists = True
                                gopro_segment = gopro[str(i)]
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
                                bit_lonlat = gm_segment_lonlat[start: end]

                                aran_match_start, aran_match_end = find_best_start_and_end_indeces_by_lonlat(aran_segment_lonlat, bit_lonlat)
                                aran_bit_data, aran_bit_attributes = extract_bit_data(aran_segment, aran_match_start, aran_match_end)
                                aran_dataset = time_subgroup.create_dataset("aran", data=aran_bit_data)
                                aran_dataset.attrs.update(aran_bit_attributes)
                                aran_counts.append(aran_match_end - aran_match_start)

                                p79_match_start, p79_match_end = find_best_start_and_end_indeces_by_lonlat(p79_segment_lonlat, bit_lonlat)
                                p79_bit_data, p79_bit_attributes = extract_bit_data(p79_segment, p79_match_start, p79_match_end)
                                p79_dataset = time_subgroup.create_dataset("p79", data=p79_bit_data)
                                p79_dataset.attrs.update(p79_bit_attributes)
                                p79_counts.append(p79_match_end - p79_match_start)

                                if verbose and (aran_counts[-1] < 3 or p79_counts[-1] < 3):
                                    # Plot the longitude and lattitude coordinates of the gm segment and the matched ARAN and P79 data
                                    verbose_resample_plot(bit_lonlat, aran_segment_lonlat, (aran_match_start, aran_match_end), p79_segment_lonlat, (p79_match_start, p79_match_end))
    
    
    if verbose:
        # Plot the segment length distributions for ARAN and P79
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

def verbose_resample_plot(bit_lonlat: np.ndarray, aran_segment_lonlat: np.ndarray, aran_match_bit: tuple[int, int], p79_segment_lonlat: np.ndarray, p79_match_bit: tuple[int, int]) -> None:
    """ Plot the longitude and lattitude coordinates of the gm segment and the matched ARAN and P79 data """

    fig, ax = plt.subplots()
    ax.plot(bit_lonlat[:, 0], bit_lonlat[:, 1], label='GM', c='k')
    # Extract ARAN and P79 with n extra points on each side for better visualization
    n = 3
    aran_match_before = max(0, aran_match_bit[0] - n)
    aran_match_after = min(len(aran_segment_lonlat), aran_match_bit[1] + n)
    wider_aran_bit = aran_segment_lonlat[aran_match_before: aran_match_after]
    shallow_aran_bit = aran_segment_lonlat[aran_match_bit[0]: aran_match_bit[1]]
    ax.plot(wider_aran_bit[:, 0], wider_aran_bit[:, 1], label='ARAN', linestyle='--', alpha=0.5)
    ax.scatter(shallow_aran_bit[:, 0], shallow_aran_bit[:, 1], alpha=0.5, marker='x')
    if aran_match_before < aran_match_bit[0]:
        ax.scatter(aran_segment_lonlat[aran_match_before: aran_match_bit[0], 0], aran_segment_lonlat[aran_match_before: aran_match_bit[0], 1], alpha=0.5, marker='x', c='r')
    if aran_match_after > aran_match_bit[1]:
        ax.scatter(aran_segment_lonlat[aran_match_bit[1]: aran_match_after, 0], aran_segment_lonlat[aran_match_bit[1]: aran_match_after, 1], alpha=0.5, marker='x', c='r')

    p79_match_before = max(0, p79_match_bit[0] - n)
    p79_match_after = min(len(p79_segment_lonlat), p79_match_bit[1] + n)
    wider_p79_bit = p79_segment_lonlat[p79_match_before: p79_match_after]
    shallow_p79_bit = p79_segment_lonlat[p79_match_bit[0]: p79_match_bit[1]]
    ax.plot(wider_p79_bit[:, 0], wider_p79_bit[:, 1], label='P79', linestyle='--', alpha=0.5)
    ax.scatter(shallow_p79_bit[:, 0], shallow_p79_bit[:, 1], alpha=0.5, marker='x')
    if p79_match_before < p79_match_bit[0]:
        ax.scatter(p79_segment_lonlat[p79_match_before: p79_match_bit[0], 0], p79_segment_lonlat[p79_match_before: p79_match_bit[0], 1], alpha=0.5, marker='x', c='r')
    if p79_match_after > p79_match_bit[1]:
        ax.scatter(p79_segment_lonlat[p79_match_bit[1]: p79_match_after, 0], p79_segment_lonlat[p79_match_bit[1]: p79_match_after, 1], alpha=0.5, marker='x', c='r')
    # Place legend outside of plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()