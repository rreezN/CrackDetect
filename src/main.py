from argparse import ArgumentParser

from src.data.make_dataset import main as make_dataset





def get_make_dataset_args():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=['convert', 'validate', 'segment', 'match', 'resample', 'kpi', 'all'], help='Mode to run the script in (all runs all modes in sequence)')
    parser.add_argument('--begin-from', action='store_true', help='Start from specified mode (inclusive)')
    parser.add_argument('--skip-gopro', action='store_true', help='Skip GoPro data in all steps')
    parser.add_argument('--speed-threshold', type=int, default=5, help='Speed threshold for segmenting data (km/h)')
    parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for segmenting data')
    parser.add_argument('--validation-threshold', type=float, default=0.2, help='Normalised MSE threshold for validating data')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    return parser.parse_args()


def get_feature_extraction_args():
    parser = ArgumentParser
    
    return parser.parse_args()

if __name__ == '__main__':
    make_dataset_args = get_make_dataset_args()
    make_dataset_args.mode = 'all'
    make_dataset(make_dataset_args)