import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .hdf5_utils import save_hdf5


# ========================================================================================================================
#           Matching functions
# ========================================================================================================================

def match_data(
        aran_hh: str = "data/raw/ref_data/cph1_aran_hh.csv",
        aran_vh: str = "data/raw/ref_data/cph1_aran_vh.csv",
        p79_hh: str = "data/raw/ref_data/cph1_zp_hh.csv",
        p79_vh: str = "data/raw/ref_data/cph1_zp_vh.csv",
        p79_iri_rut_hh: str = "data/raw/ref_data/cph1_iri_mpd_rut_hh.csv",
        p79_iri_rut_vh: str = "data/raw/ref_data/cph1_iri_mpd_rut_vh.csv",
        skip_gopro: bool = False
        ) -> None:
    """
    Match the AutoPi data with the reference data (ARAN and P79) and the GoPro data into segments

    Parameters
    ----------
    aran_hh : str
        The path to the ARAN data for the HH direction
    aran_vh : str
        The path to the ARAN data for the VH direction
    p79_hh : str
        The path to the P79 data for the HH direction
    p79_vh : str
        The path to the P79 data for the VH direction
    p79_iri_rut_hh : str
        The path to the P79 IRI and RUT data for the HH direction
    p79_iri_rut_vh : str
        The path to the P79 IRI and RUT data for the VH direction
    skip_gopro : bool
        Whether to skip the GoPro data or not
    """
    if not isinstance(aran_hh, str):
        raise TypeError(f"Input value 'aran_hh' type is {type(aran_hh)}, but expected str.")
    if not Path(aran_hh).exists():
        raise FileNotFoundError(f"Path '{aran_hh}' does not exist.")
    if not Path(aran_hh).suffix == '.csv':
        raise ValueError(f"File '{aran_hh}' is not a csv file. Expected .csv file.")
    
    if not isinstance(aran_vh, str):
        raise TypeError(f"Input value 'aran_vh' type is {type(aran_vh)}, but expected str.")
    if not Path(aran_vh).exists():
        raise FileNotFoundError(f"Path '{aran_vh}' does not exist.")
    if not Path(aran_vh).suffix == '.csv':
        raise ValueError(f"File '{aran_vh}' is not a csv file. Expected .csv file.")

    if not isinstance(p79_hh, str):
        raise TypeError(f"Input value 'p79_hh' type is {type(p79_hh)}, but expected str.")
    if not Path(p79_hh).exists():
        raise FileNotFoundError(f"Path '{p79_hh}' does not exist.")
    if not Path(p79_hh).suffix == '.csv':
        raise ValueError(f"File '{p79_hh}' is not a csv file. Expected .csv file.")

    if not isinstance(p79_vh, str):
        raise TypeError(f"Input value 'p79_vh' type is {type(p79_vh)}, but expected str.")
    if not Path(p79_vh).exists():
        raise FileNotFoundError(f"Path '{p79_vh}' does not exist.")
    if not Path(p79_vh).suffix == '.csv':
        raise ValueError(f"File '{p79_vh}' is not a csv file. Expected .csv file.")

    if not isinstance(p79_iri_rut_hh, str):
        raise TypeError(f"Input value 'p79_iri_rut_hh' type is {type(p79_iri_rut_hh)}, but expected str.")
    if not Path(p79_iri_rut_hh).exists():
        raise FileNotFoundError(f"Path '{p79_iri_rut_hh}' does not exist.")
    if not Path(p79_iri_rut_hh).suffix == '.csv':
        raise ValueError(f"File '{p79_iri_rut_hh}' is not a csv file. Expected .csv file.")
    
    if not isinstance(p79_iri_rut_vh, str):
        raise TypeError(f"Input value 'p79_iri_rut_vh' type is {type(p79_iri_rut_vh)}, but expected str.")
    if not Path(p79_iri_rut_vh).exists():
        raise FileNotFoundError(f"Path '{p79_iri_rut_vh}' does not exist.")
    if not Path(p79_iri_rut_vh).suffix == '.csv':
        raise ValueError(f"File '{p79_iri_rut_vh}' is not a csv file. Expected .csv file.")
    

    if not isinstance(skip_gopro, bool):
        raise TypeError(f"Input value 'skip_gopro' type is {type(skip_gopro)}, but expected bool.")

    prefix = aran_hh.split("data/")[0]

    # Define path to segment files
    segment_file = prefix + 'data/interim/gm/segments.hdf5'

    # Load reference and GoPro data
    aran = {
        'hh': pd.read_csv(aran_hh, sep=';', encoding='unicode_escape').fillna(0),
        'vh': pd.read_csv(aran_vh, sep=';', encoding='unicode_escape').fillna(0)
    }

    p79 = {
        'hh': pd.read_csv(p79_hh, sep=';', encoding='unicode_escape'),
        'vh': pd.read_csv(p79_vh, sep=';', encoding='unicode_escape')
    }


    p79_iri_rut = {
        'hh': pd.read_csv(p79_iri_rut_hh, sep=';', encoding='unicode_escape'),
        'vh': pd.read_csv(p79_iri_rut_vh, sep=';', encoding='unicode_escape')
    }

    # Now we combine the IRI and RUT information from p79_iri_rut into the p79 dataframes (used for KPIs)
    for direction in ['hh', 'vh']:
        iri_rut_array = np.zeros((len(p79[direction]), 4))
        # dataframe with 4 columns: IRI (5), IRI (21), Venstre sporkøring, Højre sporkøring
        iri_rut_df = pd.DataFrame(iri_rut_array, columns=['IRI (5) [m/km]', 'IRI (21) [m/km]', 'Venstre sporkøring [mm]', 'Højre sporkøring [mm]'])
        for idx, row in (pbar := tqdm(p79_iri_rut[direction].iterrows())):
            pbar.set_description(f"Combining IRI and RUT data for {direction} direction ({idx+1}/{len(p79_iri_rut[direction])}")
            distance = row['Distance [m]']
            mask = (p79[direction]['Distance [m]'] >= distance) & (p79[direction]['Distance [m]'] < distance+10)
            iri_rut_df.loc[mask, 'IRI (5) [m/km]'] = row[' IRI (5) [m/km]']
            iri_rut_df.loc[mask, 'IRI (21) [m/km]'] = row[' IRI (21) [m/km]']
            iri_rut_df.loc[mask, 'Venstre sporkøring [mm]'] = row[' Venstre sporkøring [mm]']
            iri_rut_df.loc[mask, 'Højre sporkøring [mm]'] = row[' Højre sporkøring [mm]']
        p79[direction] = pd.concat([p79[direction], iri_rut_df], axis=1)

    # Create folders for saving
    Path(prefix + 'data/interim/aran').mkdir(parents=True, exist_ok=True)
    Path(prefix + 'data/interim/p79').mkdir(parents=True, exist_ok=True)
    folders = ["aran", "p79"]

    if not skip_gopro:
        # Load GoPro data
        gopro_data = {}
        car_trips = ["16011", "16009", "16006"]
        for trip_id in car_trips:
            gopro_data[trip_id] = {}
            for measurement in ['gps5', 'accl', 'gyro']:
                gopro_data[trip_id][measurement] = pd.read_csv(prefix + f'data/interim/gopro/{trip_id}/{measurement}.csv')
        Path(prefix + 'data/interim/gopro').mkdir(parents=True, exist_ok=True)
        folders.append("gopro")

    # Remove old segment files if they exist
    for folder in folders:
        segment_path = Path(prefix + f'data/interim/{folder}/segments.hdf5')
        if segment_path.exists():
            segment_path.unlink()

    # Match data
    with h5py.File(segment_file, 'r') as f:
        segment_files = [f[str(i)] for i in range(len(f))]
        pbar = tqdm(segment_files)
        for i, segment in enumerate(pbar):
            pbar.set_description(f"Matching segment {i+1:03d}/{len(segment_files)}")

            direction = segment.attrs['direction']
            trip_name = segment.attrs["trip_name"]

            segment_lonlat = segment['gps'][()][:, 2:0:-1]  # ['measurements]

            aran_dir = aran[direction]
            p79_dir = p79[direction]

            # Match to ARAN data
            aran_match = find_best_start_and_end_indeces_by_lonlat(aran_dir[["Lon", "Lat"]].values, segment_lonlat)
            aran_segment = cut_dataframe_by_indices(aran_dir, *aran_match)
            save_hdf5(aran_segment, prefix + 'data/interim/aran/segments.hdf5', segment_id=i, attributes=segment.attrs)

            # Match to P79 data
            p79_match = find_best_start_and_end_indeces_by_lonlat(p79_dir[["Lon", "Lat"]].values, segment_lonlat)
            p79_segment = cut_dataframe_by_indices(p79_dir, *p79_match)
            save_hdf5(p79_segment, prefix + 'data/interim/p79/segments.hdf5', segment_id=i, attributes=segment.attrs)
            
            if skip_gopro:
                continue
            
            # gopro is a little different.. (These trips do not have any corresponding gopro data, so we skip them)
            if trip_name not in ["16006", "16009", "16011"]:
                continue
            
            gopro_segment = {}
            for measurement in ['gps5', 'accl', 'gyro']:
                start_index, end_index, start_diff, end_diff = find_best_start_and_end_indeces_by_time(segment["gps"][()], gopro_data[trip_name][measurement]["date"])

                if max(start_diff, end_diff) > 1:
                    continue
                
                gopro_segment[measurement] = gopro_data[trip_name][measurement][start_index:end_index].to_dict('series')

            if gopro_segment != {}:
                save_hdf5(gopro_segment, prefix + 'data/interim/gopro/segments.hdf5', segment_id=i, attributes=segment.attrs)


def find_best_start_and_end_indeces_by_lonlat(trip: np.ndarray, section: np.ndarray) -> tuple[int, int]:
    """
    Find the start and end indeces of the section data that are closest to the trip data

    Parameters
    ----------
    trip : np.ndarray
        The longitudal and lattitudal coordinates of the trip data
    section : np.ndarray
        The longitudal and lattitudal coordinates of the section

    Returns
    -------
    tuple[int, int]
        The start and end indeces of the section data
    """
    if not isinstance(trip, np.ndarray):
        raise TypeError(f"Input value 'trip' type is {type(trip)}, but expected np.ndarray.")
    if not isinstance(section, np.ndarray):
        raise TypeError(f"Input value 'section' type is {type(section)}, but expected np.ndarray.")
    # Find the start and end indeces of the section data that are closest to the trip data
    lon_a, lat_a = trip[:,0], trip[:,1]
    lon_b, lat_b = section[:,0], section[:,1]
    
    start_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[0], lat_b[0]]), axis=1))
    end_index = np.argmin(np.linalg.norm(np.column_stack((lon_a, lat_a)) - np.array([lon_b[-1], lat_b[-1]]), axis=1))

    return start_index.item(), end_index.item()+1


def find_best_start_and_end_indeces_by_time(current_segment_time: np.ndarray, gopro_time: pd.Series) -> tuple[int, int, float, float]:
    """
    Find the start and end indeces of the section data based on time

    Parameters
    ----------
    current_segment : np.ndarray
        The time data from the current segment
    gopro_time : pd.Series
        The time data from the GoPro

    Returns
    -------
    tuple[int, int, float, float]
        The start and end indeces of the section data based on time
    """
    if not isinstance(current_segment_time, np.ndarray):
        raise TypeError(f"Input value 'current_segment_time' type is {type(current_segment_time)}, but expected np.ndarray.")
    if not isinstance(gopro_time, pd.Series):
        raise TypeError(f"Input value 'gopro_time' type is {type(gopro_time)}, but expected pd.Series.")
    # Find the start and end indeces of the section data based on time
    current_segment_start_time = current_segment_time[0, 0]
    current_segment_end_time = current_segment_time[-1, 0]
    segment_time = [current_segment_start_time, current_segment_end_time]
    
    diff_start = (gopro_time - segment_time[0]).abs()
    start_index = diff_start.idxmin()
    start_diff = diff_start.min()
    
    diff_end = (gopro_time - segment_time[1]).abs()
    end_index = diff_end.idxmin()
    end_diff = diff_end.min()

    return start_index, end_index, start_diff, end_diff


def cut_dataframe_by_indices(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Cut the dataframe by the start and end indeces

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to cut
    start : int
        The start index
    end : int
        The end index
    
    Returns
    -------
    pd.DataFrame
        The cut dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input value 'df' type is {type(df)}, but expected pd.DataFrame.")
    if not isinstance(start, int):
        raise TypeError(f"Input value 'start' type is {type(start)}, but expected int.")
    if not isinstance(end, int):
        raise TypeError(f"Input value 'end' type is {type(end)}, but expected int.")
    # Cut the dataframe by the start and end indeces
    return df.iloc[start:end]
