from pathlib import Path
from tqdm import tqdm

from .hdf5_utils import save_hdf5, unpack_hdf5


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

    Returns
    -------
    int
        The new segment index
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


def segment_gm_trip(measurements: dict, trip_name: str, pass_name: str, direction: str, speed_threshold: int = 5, time_threshold: int = 10) -> list[dict]:
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

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing the measurements for a section of the trip
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