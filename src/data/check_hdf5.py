import h5py
import numpy as np
from argparse import ArgumentParser


def delete_model(file_path: str, keys: list[str]):
    """Delets models from the hdf5 file.

    Parameters:
    ----------
        file_path (str): The file to delete the models from.
        keys (list[str]): A list of models to delete. Each model is a key in the hdf5 file. ['model1', 'model2', ...]
    """
    
    with h5py.File(file_path, 'r+') as f:
        data_types = ['train', 'test', 'val']
        for data_type in data_types:
            for segment in f[data_type]['segments'].keys():
                for second in f[data_type]['segments'][segment].keys():
                    for key in keys:
                        if key in f[data_type]['segments'][segment][second].keys():
                            del f[data_type]['segments'][segment][second][key]
                            print(f'Deleted {key} from {data_type}/{segment}/{second}')
            


def check_hdf5(file_path: str):
    """Prints the structure of the hdf5 file, with the option to print a summary of the data.

    Parameters:
    ----------
        file_path (str): path to the hdf5 file.
    """
    
    print('\n    ---### HDF5 FILE STRUCTURE ###---\n')
    
    with h5py.File(file_path, 'r') as f:
        
        for key1 in f.keys():
            print(f'- {key1}')
            
            for key2 in f[key1].keys():
                print(f'  - {key2}')
                
                for i, key3 in enumerate(f[key1][key2].keys()):
                    print(f'    - {key3}')
                    
                    for j, key4 in enumerate(f[key1][key2][key3].keys()):
                        print(f'      - {key4}')
                        
                        if isinstance(f[key1][key2][key3][key4], h5py.Dataset):
                            print(f'        - {key4}')
                        
                        else:
                            for k, key5 in enumerate(f[key1][key2][key3][key4].keys()):
                                print(f'        - {key5}')
                            
                        if j > 1:
                            print('        ...')
                            break
                    
                    if i > args.limit:
                        print('    ...')
                        break
                    
        
        
        if not args.no_summary:
            print('\nCalculating summary...')
            nr_segments = len(f['train']['segments'].keys()) + len(f['test']['segments'].keys()) + len(f['val']['segments'].keys())
            nr_seconds = 0
            
            for data_type in ['train', 'test', 'val']:
                for key in f[data_type]['segments'].keys():
                    nr_seconds += len(f['train']['segments'][key].keys())
            
            first_segment = list(f['train']['segments'].keys())[0]
            first_second = list(f['train']['segments'][first_segment].keys())[0]
            nr_models = len(f['train']['segments'][first_segment][first_second].keys()) - 1
            
            print('\n    ---### SUMMARY ###---\n')
            print(f'Segments: {nr_segments}')
            print(f'Seconds: {nr_seconds}')
            print(f'Models: {nr_models}')
            models = []
            for model in f['train']['segments'][first_segment][first_second].keys():
                if model != 'kpis':
                    print(f'  - {model}: {f["train"]["segments"][first_segment][first_second][model].shape}')
                    models += [model]
            print(f'\nStatistics:')
            for model in models:
                print(f'  - {model}')
                for key in f['train']['statistics'][model[:-1]].keys():
                    data = f['train']['statistics'][model[:-1]][key][()]
                    if isinstance(data[0], np.float32) or isinstance(data[0], np.int32) or isinstance(data[0], np.float64) or isinstance(data[0], np.int64):
                        data = np.round(np.array(data), 3)
                    print(f'    - {key}: {data}')
            print(f'  - KPIs:')
            for key in f['train']['statistics']['kpis']['1'].keys():
                data = f['train']['statistics']['kpis']['1'][key][()]
                if isinstance(data[0], np.float32) or isinstance(data[0], np.int32) or isinstance(data[0], np.float64) or isinstance(data[0], np.int64):
                    data = np.round(np.array(data), 3)
                print(f'    - {key}: {data}')
            
            print()
        
                            
            


def get_args():
    parser = ArgumentParser(description='Check hdf5 file')
    parser.add_argument('--file_path', type=str, default='data/processed/features.hdf5')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--no_summary', action='store_true')
    parser.add_argument('--delete_model', type=list[str], nargs='+', default=[])
    return parser.parse_args()
            
if __name__ == '__main__':
    args = get_args()
    
    if args.delete_model != []:
        delete_model(args.file_path, args.delete)
    
    check_hdf5(args.file_path)
    
    