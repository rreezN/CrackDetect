import h5py
from pathlib import Path
from tqdm import tqdm

from .hdf5_utils import save_hdf5


# ========================================================================================================================
#           Segmentation functions
# ========================================================================================================================

def segment_gm(autopi: h5py.Group, direction: str, prefix : str, speed_threshold: float = 5, time_threshold: float = 10, segment_index: int = 0) -> int:
    """
    Segment the GM data into sections where the vehicle is moving
    
    Parameters
    ----------
    autopi : h5py.Group
        The AutoPi data group
    direction : str
        The direction of the trip, either 'hh' or 'vh'
    speed_threshold : float
        The speed in km/h below which the vehicle is considered to be stopped
    time_threshold : float
        The minimum time in seconds for a section to be considered valid
    segment_index : int
        The index to start the segment numbering from

    Returns
    -------
    int
        The new segment index
    """
    if not isinstance(autopi, h5py.Group):
        raise TypeError(f"Input value 'autopi' type is {type(autopi)}, but expected h5py.Group.")
    if direction not in ['hh', 'vh']:
        raise ValueError(f"Input value 'direction' is {direction}, but expected 'hh' or 'vh'.")
    if not isinstance(speed_threshold, float | int):
        raise TypeError(f"Input value 'speed_threshold' type is {type(speed_threshold)}, but expected float or int.")
    if not isinstance(time_threshold, float | int):
        raise TypeError(f"Input value 'time_threshold' type is {type(time_threshold)}, but expected float or int.")
    if not isinstance(segment_index, int):
        raise TypeError(f"Input value 'segment_index' type is {type(segment_index)}, but expected int.")

    # direction is either 'hh' or 'vh'
    pbar = tqdm(autopi.items())
    for trip_name, trip in pbar:
        for pass_name, pass_ in trip.items():
            pbar.set_description(f"Interpolating {trip_name} {pass_name}")
            segments, attributes = segment_gm_trip(pass_, trip_name, pass_name, direction, speed_threshold=speed_threshold, time_threshold=time_threshold)
            for segment, attribute in zip(segments, attributes):
                # Save the segment dictionary to a hdf5 file
                segment_path = Path(prefix + f'data/interim/gm/segments.hdf5')
                save_hdf5(segment, segment_path, segment_id=segment_index, attributes=attribute)

                # Increment the segment index
                segment_index += 1
    return segment_index


def segment_gm_trip(measurements: h5py.Group, trip_name: str, pass_name: str, direction: str, speed_threshold: float = 5, time_threshold: float = 10) -> tuple[list[dict], list[dict]]:
    """
    Segment a single GM trip into sections where the vehicle is moving
    
    Parameters
    ----------
    measurements : h5py.Group
        The autopi pass group
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
    if not isinstance(measurements, h5py.Group):
        raise TypeError(f"Input value 'measurements' type is {type(measurements)}, but expected h5py.Group.")
    if not isinstance(trip_name, str):
        raise TypeError(f"Input value 'trip_name' type is {type(trip_name)}, but expected str.")
    if not isinstance(pass_name, str):
        raise TypeError(f"Input value 'pass_name' type is {type(pass_name)}, but exprected str.")
    if direction not in ['hh', 'vh']:
        raise ValueError(f"Input value 'direction' is {direction}, but expected 'hh' or 'vh'.")
    if not isinstance(speed_threshold, float | int):
        raise TypeError(f"Input value 'speed_threshold' type is {type(speed_threshold)}, but expected float or int.")
    if not isinstance(time_threshold, float | int):
        raise TypeError(f"Input value 'time_threshold' type is {type(time_threshold)}, but expected float or int.")

    # Find the ranges in time where the speed is not zero
    non_zero_speed_ranges = []
    start_index = -1
    for i in range(len(measurements["spd_veh"][()])):
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
    attributes = []
    for start, end in non_zero_speed_ranges:
        # Check if the section is too short
        if end - start < time_threshold:
            continue
        attribute = {
            "trip_name": trip_name,
            "pass_name": pass_name,
            "direction": direction
        }
        section = {
        }
        for key, value in measurements.items():
            section[key] = value[(value[:, 0] >= start) & (value[:, 0] <= end)]
        sections.append(section)
        attributes.append(attribute)
    
    return sections, attributes

def segment(hh: str = 'data/interim/gm/converted_platoon_CPH1_HH.hdf5', vh : str = 'data/interim/gm/converted_platoon_CPH1_VH.hdf5', speed_threshold: float = 5, time_threshold: float = 10) -> None:
    """
    Segment the GM data into sections where the vehicle is moving.

    Parameters
    ----------
    speed_threshold : int
        The speed in km/h below which the vehicle is considered to be stopped
    time_threshold : int
        The minimum time in seconds for a section to be considered valid
    """
    if not isinstance(hh, str):
        raise TypeError(f"Input value 'hh' type is {type(hh)}, but expected str.")
    assert Path(hh).exists(), f"Path '{hh}' does not exist."
    if not isinstance(vh, str):
        raise TypeError(f"Input value 'vh' type is {type(vh)}, but expected str.")
    assert Path(vh).exists(), f"Path '{vh}' does not exist."
    if not isinstance(speed_threshold, float | int):
        raise TypeError(f"Input value 'speed_threshold' type is {type(speed_threshold)}, but expected float or int.")
    if not isinstance(time_threshold, float | int):
        raise TypeError(f"Input value 'time_threshold' type is {type(time_threshold)}, but expected float or int.")
    

    prefix = hh.split("data/")[0]

    # Remove old segment file if it exists
    segment_path = Path(prefix + 'data/interim/gm/segments.hdf5')
    if segment_path.exists():
        segment_path.unlink()

    # Segment data
    with h5py.File(hh, 'r') as autopi_hh:
        segment_index = segment_gm(autopi_hh['GM'], direction='hh', prefix=prefix, speed_threshold=speed_threshold, time_threshold=time_threshold)
    
    with h5py.File(vh, 'r') as autopi_vh:
        segment_gm(autopi_vh['GM'], direction='vh', prefix = prefix, speed_threshold=speed_threshold, time_threshold=time_threshold, segment_index=segment_index)