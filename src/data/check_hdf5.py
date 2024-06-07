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


def h5_tree(val, pre: str = '', max_items: int = 10):
    """Recursively print the hdf5 file structure.

    Parameters:
    -----------
        val (h5py.File): The hdf5 file.
        pre (str, optional): The pre-fix of the file (determines the level to call the function at). Defaults to ''.
        max_items (int, optional): The maximum number of items to print at each level. Defaults to 10.
    """
    items = len(val)
    for i, (key, val) in enumerate(val.items()):
        if i >= max_items:
            print(pre + '⋮')
            break
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ', max_items=max_items)
            else:
                try:
                    print(f'{pre}  └──   {key}   {np.shape(val)}')
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ', max_items=max_items)
            else:
                try:
                    print(f'{pre}  ├──   {key}   {np.shape(val)}')
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def summary(file_path: str):
    """Print a summary of the hdf5 file.
    NOTE: Only works for features.hdf5 files created with the feature_extractor.py script.

    Parameters:
    ------------
        file_path (str): Path to the hdf5 file.
    """
    with h5py.File(file_path, 'r+') as f:
        if 'train' not in f.keys():
            raise ValueError('The hdf5 file does not contain the expected keys.')
        
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
            for key in f['train']['statistics'][model].keys():
                data = f['train']['statistics'][model][key][()]
                if isinstance(data[0], np.float32) or isinstance(data[0], np.int32) or isinstance(data[0], np.float64) or isinstance(data[0], np.int64):
                    data = np.round(np.array(data), 3)
                print(f'    - {key}: {data}, shape: {data.shape}')
        print(f'  - KPIs:')
        for key in f['train']['statistics']['kpis']['1'].keys():
            data = f['train']['statistics']['kpis']['1'][key][()]
            if isinstance(data[0], np.float32) or isinstance(data[0], np.int32) or isinstance(data[0], np.float64) or isinstance(data[0], np.int64):
                data = np.round(np.array(data), 3)
            print(f'    - {key}: {data}, shape: {data.shape}')
        
        print()                  


def get_args():
    parser = ArgumentParser(description='Check the hdf5 file structure.')
    parser.add_argument('--file_path', type=str, default='data/processed/features.hdf5', help='The path to the hdf5 file.')
    parser.add_argument('--limit', type=int, default=5, help='The maximum number of items to print at each level.')
    parser.add_argument('--summary', action='store_true', help='Print a summary of the hdf5 file. NOTE: Only works for features.hdf5 files created with the feature_extractor.py script.')
    parser.add_argument('--delete_model', nargs='+', default=[], help='Delete models from the hdf5 file, example: --delete_model model1 model2')
    return parser.parse_args()
            
if __name__ == '__main__':
    args = get_args()
    
    if args.delete_model != []:
        delete_model(args.file_path, args.delete_model)
    
    print('\n    ---### HDF5 FILE STRUCTURE ###---\n')
    h5_tree(val=h5py.File(args.file_path, 'r'), max_items=args.limit)
    
    if args.summary:
        summary(args.file_path)
    