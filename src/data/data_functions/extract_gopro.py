import datetime as dt
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# ========================================================================================================================
#           Hardcoded GoPro functions
# ========================================================================================================================

def csv_files_together(car_trip: str, go_pro_names: list[str], car_number: str, raw_folder: str = "data/raw/gopro") -> None:
    """
    Saves the GoPro data to a csv file for each trip

    Parameters
    ----------
    car_trip : str
        The trip name
    go_pro_names : list[str]
        The names of the GoPro cameras
    car_number : str
        The car number
    """
    # assert
    assert type(car_trip) == str, f"Input value 'car_trip' type is {type(car_trip)}, but expected str."
    assert type(go_pro_names) == list, f"Input value 'go_pro_names' type is {type(go_pro_names)}, but expected list."
    assert type(car_number) == str, f"Input value 'car_number' type is {type(car_number)}, but expected str."
    assert type(raw_folder) == str, f"Input value 'raw_folder' type is {type(raw_folder)}, but expected str."
    assert Path(raw_folder).exists(), f"Path '{raw_folder}' does not exist. Ensure gopro data is in the correct folder structure."
    
    
    # Load all the gopro data 
    for measurement in ['accl', 'gps5', 'gyro']:
        gopro_data = None
        for trip_id in go_pro_names:
            trip_folder =  f"{raw_folder}/{car_number}/{trip_id}"
            new_data = pd.read_csv(f'{trip_folder}/{trip_id}_HERO8 Black-{measurement.upper()}.csv')
            new_data['date'] = pd.to_datetime(new_data['date']).map(dt.datetime.timestamp)
            
            # Drop all non-float columns
            new_data = new_data.select_dtypes(include=['float64', 'float32'])
        
            if gopro_data is not None:
                gopro_data = pd.concat([gopro_data, new_data])
            else:
                gopro_data = new_data
            
        # save gopro_data[measurement]
        interim_folder = raw_folder.replace("/raw/", "/interim/")
        new_folder = f"{interim_folder}/{car_trip}"
        Path(new_folder).mkdir(parents=True, exist_ok=True)
        
        gopro_data.to_csv(f"{new_folder}/{measurement}.csv", index=False)

def preprocess_gopro_data(folder: str = "data/raw/gopro") -> None:
    """
    Preprocess the GoPro data by combining the data from the three GoPro cameras into one csv file for each trip

    NOTE: This function is hardcoded for the three trips in the CPH1 dataset
    """
    assert type(folder) == str, f"Input value 'folder' type is {type(folder)}, but expected str."
    assert Path(folder).exists(), f"Path '{folder}' does not exist. Ensure gopro data is in the correct folder structure."

    # Create gopro data for the three trips
    car_trips = ["16011", "16009", "16006"]
    car_gopro = {
        "16011": ["GH012200", "GH022200", "GH032200", "GH042200", "GH052200", "GH062200"],
        "16009": ["GH010053", "GH030053", "GH040053", "GH050053", "GH060053"],
        "16006": ["GH020056", "GH040053"]
    }
    car_numbers = {
        "16011": "car1",
        "16009": "car3",
        "16006": "car3"
    }
    
    pbar = tqdm(car_trips)
    for car_trip in pbar:
        pbar.set_description(f"Converting GoPro/{car_trip}")
        csv_files_together(car_trip, car_gopro[car_trip], car_numbers[car_trip], raw_folder=folder)