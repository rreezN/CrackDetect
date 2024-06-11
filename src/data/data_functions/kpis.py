#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path


# ========================================================================================================================
#           KPI functions
# ========================================================================================================================

def compute_kpis(segment_path: str = 'data/processed/wo_kpis/segments.hdf5', window_sizes: list[int] = [1, 2]) -> None:
    """
    Alters the existing segments file by adding KPIs to each second in each segment, based on the window sizes provided.

    Do account for the fact that the first and last seconds of each segment depend on the max(window_sizes), such that the KPIs can be computed,
    and compared across window-sizes.

    Parameters
    ----------
    segment_path : str
        The path to the segments file. Default is 'data/processed/w_kpis/segments.hdf5'.
    window_sizes : list[int]
        The window sizes to compute KPIs for. Default is [1, 2]. 
    """
    assert type(segment_path) == str, f"Input value 'segment_path' type is {type(segment_path)}, but expected str."
    assert Path(segment_path).exists(), f"Path '{segment_path}' does not exist."
    assert type(window_sizes) == list, f"Input value 'window_sizes' type is {type(window_sizes)}, but expected list."
    assert all([type(i) == int for i in window_sizes]), f"Input value 'window_sizes' contains non-integers."
    
    
    path_split = segment_path.split('w_kpis')
    if len(path_split) == 1:
        path_split = segment_path.split('wo_kpis')

    w_path = path_split[0] + 'w_kpis' + path_split[1]
    wo_path = path_split[0] + 'wo_kpis' + path_split[1]

    # Create folders for saving
    Path(path_split[0] + 'w_kpis').mkdir(parents=True, exist_ok=True)

    # Remove old segment files if they exist
    segment_path = Path(w_path)
    if segment_path.exists():
        segment_path.unlink()
    
    # Assert that data exists
    assert Path(wo_path).exists(), f"Path '{wo_path}' does not exist."
    assert Path(wo_path).suffix == '.hdf5', f"File '{wo_path}' is not a hdf5 file."
    assert Path(w_path).exists(), f"Path '{w_path}' does not exist."
    assert Path(w_path).suffix == '.hdf5', f"File '{w_path}' is not a hdf5 file."
    
    # Load processed data    
    with h5py.File(wo_path, 'r') as f:
        # Open final processed segments file
        with h5py.File(w_path, 'a') as f2:
            for i, segment in (pbar := tqdm(f.items())):
                pbar.set_description(f"Computing KPIs for segment {i}")
                segment_subgroup = f2.create_group(str(i))

                # Add direction, trip name and pass name as attr to segment subgroup
                segment_subgroup.attrs['direction'] = segment.attrs['direction']
                segment_subgroup.attrs['trip_name'] = segment.attrs['trip_name']
                segment_subgroup.attrs['pass_name'] = segment.attrs['pass_name']

                num_seconds_in_segment = len(segment)

                for j, second in segment.items():
                    j = int(j)
                    # Skip the first and last seconds which can not be computed with a window size of max(window_sizes)
                    if j < max(window_sizes) or j >= num_seconds_in_segment - max(window_sizes):
                        continue

                    second_subgroup = segment_subgroup.create_group(str(j))

                    for key, value in second.items():
                        second_subgroup.create_dataset(key, data=value[()])
                        second_subgroup[key].attrs.update(second[key].attrs)
                    
                    # Compute KPIs
                    kpi_subgroup = second_subgroup.create_group('kpis')
                    kpi_subgroup.attrs['window_sizes'] = window_sizes
                    for window_size in window_sizes:
                        kpis = compute_kpis_for_second(segment, j, window_size)
                        kpi_data = kpi_subgroup.create_dataset(str(window_size), data=kpis)
                        for i, kpi_name in enumerate(['DI', 'RUT', 'PI', 'IRI']):
                            kpi_data.attrs[kpi_name] = i


def compute_kpis_for_second(segment: h5py.Group, second_index: int, window_size: int) -> np.ndarray:
    """
    Compute KPIs for a given second in a segment, based on a window size.

        Crackingsum = (LCS^2 + LCM^3 + LCL^4 + 3*TCS + 4*TCM + 5*TCL)^0.1
        Alligatorsum = (3*ACS + 4*ACM + 5*ACL)^0.3
        Potholessum = (5*PAS + 7*PAM + 10*PAL + 5*PAD)^0.1
        
        *KPI_DI* = Crackingsum + Alligatorsum + Potholessum
    
        *KPI_RUT* = ((RDL + RDR) / 2)^0.5

        *KPI_PI* = (LCSe^2 + 2*TCSe)^0.1

        *KPI_IRI* = ((IRL + IRR) / 2)^0.2

    Names of the ARAN attributes are based on the ARAN manual,

        Live Road Assessment based on modern car sensors (LiRA): Practical guide
        by Asmus Skar et al. (2022)
        

    Parameters
    ----------
    segment : h5py.Group
        The segment to compute KPIs for.
    second_index : int
        The index of the second to compute KPIs for.
    window_size : int
        The window size to compute KPIs for.

    Returns
    -------
    np.ndarray
        The KPIs for the given second.
    """
    assert type(segment) == h5py.Group, f"Input value 'segment' type is {type(segment)}, but expected h5py.Group."
    assert type(second_index) == int, f"Input value 'second_index' type is {type(second_index)}, but expected int."
    assert type(window_size) == int, f"Input value 'window_size' type is {type(window_size)}, but expected int."
    
    
    # Extract ARAN data for all seconds within the window
    windowed_aran_data = []
    for i in range(second_index - window_size, second_index + window_size + 1):
        windowed_aran_data.append(segment[str(i)]['aran'][()])
    
    # Define aran attributes for KPI-functions
    aran_attrs = segment[str(second_index)]['aran'].attrs

    # Stack the ARAN data
    windowed_aran_data = np.vstack(windowed_aran_data)

    # Compute KPIs
    # damage index
    KPI_DI = damage_index(windowed_aran_data, aran_attrs)
    # rutting index
    KPI_RUT = rutting_mean(windowed_aran_data, aran_attrs) 
    # patching index
    PI = patching_sum(windowed_aran_data, aran_attrs)
    # IRI
    IRI = iri_mean(windowed_aran_data, aran_attrs)
    
    return np.asarray([KPI_DI, KPI_RUT, PI, IRI])


def damage_index(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Calculates the damage index for a given window of ARAN data as specified in the paper. TODO add reference to paper

    crackingsum = (LCS^2 + LCM^3 + LCL^4 + 3*TCS + 4*TCM + 5*TCL)^0.1
    alligatorsum = (3*ACS + 4*ACM + 5*ACL)^0.3
    potholessum = (5*PAS + 7*PAM + 10*PAL + 5*PAD)^0.1
    DI = crackingsum + alligatorsum + potholessum


    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The damage index for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    

    crackingsum = cracking_sum(windowed_aran_data, aran_attrs)
    alligatorsum = alligator_sum(windowed_aran_data, aran_attrs)
    potholessum = pothole_sum(windowed_aran_data, aran_attrs)
    DI = crackingsum + alligatorsum + potholessum
    return DI


def cracking_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Conventional/longitudinal and transverse cracks are reported as length.

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The cracking sum for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    LCS = windowed_aran_data[:, aran_attrs['Revner På Langs Små (m)']]
    LCM = windowed_aran_data[:, aran_attrs['Revner På Langs Middelstore (m)']]
    LCL = windowed_aran_data[:, aran_attrs['Revner På Langs Store (m)']]
    TCS = windowed_aran_data[:, aran_attrs['Transverse Low (m)']]
    TCM = windowed_aran_data[:, aran_attrs['Transverse Medium (m)']]
    TCL = windowed_aran_data[:, aran_attrs['Transverse High (m)']]
    return ((LCS**2 + LCM**3 + LCL**4 + 3*TCS + 4*TCM + 5*TCL)**(0.1)).mean()


def alligator_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Alligator cracks are computed as area of the pavement affected by the damage

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.
    
    Returns
    -------
    float
        The alligator sum for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    ACS = windowed_aran_data[:, aran_attrs['Krakeleringer Små (m²)']]
    ACM = windowed_aran_data[:, aran_attrs['Krakeleringer Middelstore (m²)']]
    ACL = windowed_aran_data[:, aran_attrs['Krakeleringer Store (m²)']]
    return ((3*ACS + 4*ACM + 5*ACL)**(0.3)).mean()


def pothole_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    Potholes are computed as the average weighted depth of the potholes based on the ARAN manual.

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The pothole sum for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    PAS = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Low (mm)']]
    PAM = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Medium (mm)']]
    PAL = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth High (mm)']]
    PAD = windowed_aran_data[:, aran_attrs['Slaghuller Max Depth Delamination (mm)']]
    return ((5*PAS + 7*PAM +10*PAL +5*PAD)**(0.1)).mean()


def rutting_mean(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager, rut: str ='straight-edge') -> float:
    """
    The rutting index is computed as the average of the square root of the rut depth for each wheel track.

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The rutting mean for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    # TODO: FIGURE OUT WHICH ONE TO USE
    if rut == 'straight-edge':
        RDL = windowed_aran_data[:, aran_attrs['LRUT Straight Edge (mm)']]
        RDR = windowed_aran_data[:, aran_attrs['RRUT Straight Edge (mm)']]
    elif rut == 'wire':
        RDL = windowed_aran_data[:, aran_attrs['LRUT Wire (mm)']]
        RDR = windowed_aran_data[:, aran_attrs['RRUT Wire (mm)']]
    return (((RDL +RDR)/2)**(0.5)).mean()


def iri_mean(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    The IRI is computed as the average of the square root of the IRI for the left and right wheel tracks.

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The IRI mean for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    IRL = windowed_aran_data[:, aran_attrs['Venstre IRI (m_km)']]
    IRR = windowed_aran_data[:, aran_attrs['Højre IRI (m_km)']]
    return (((IRL + IRR)/2)**(0.2)).mean()
    
def patching_sum(windowed_aran_data: np.ndarray, aran_attrs: h5py._hl.attrs.AttributeManager) -> float:
    """
    The patching index is computed based on the ARAN manual.    

    Parameters
    ----------
    windowed_aran_data : np.ndarray
        The ARAN data for the window.
    aran_attrs : h5py._hl.attrs.AttributeManager
        The ARAN attributes for the data.

    Returns
    -------
    float
        The patching sum for the given window.
    """
    assert type(windowed_aran_data) == np.ndarray, f"Input value 'windowed_aran_data' type is {type(windowed_aran_data)}, but expected np.ndarray."
    assert type(aran_attrs) == h5py._hl.attrs.AttributeManager, f"Input value 'aran_attrs' type is {type(aran_attrs)}, but expected h5py._hl.attrs.AttributeManager."
    
    LCSe = windowed_aran_data[:, aran_attrs['Revner På Langs Sealed (m)']]
    TCSe = windowed_aran_data[:, aran_attrs['Transverse Sealed (m)']]
    return ((LCSe**2 + 2*TCSe)**(0.1)).mean()