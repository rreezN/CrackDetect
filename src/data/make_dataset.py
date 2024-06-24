from __init__ import *
from argparse import ArgumentParser, Namespace

from src.data.data_functions.converting import convert
from src.data.data_functions.extract_gopro import preprocess_gopro_data
from src.data.data_functions.validating import validate
from src.data.data_functions.segmenting import segment
from src.data.data_functions.matching import match_data
from src.data.data_functions.resampling import resample
from src.data.data_functions.kpis import compute_kpis


def main(args: Namespace) -> None:

    begun = False

    if begun or args.mode in ['convert', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---### Converting data ###---')
        convert()
        if not args.skip_gopro:
            print('    ---### Extracting GoPro data ###---')
            # Convert GoPro data to align with the GM trips
            preprocess_gopro_data()
    
    if begun or args.mode in ['validate', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---### Validating data ###---')
        validate(threshold=args.validation_threshold, verbose=args.verbose)

    if begun or args.mode in ['segment', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---### Segmenting data ###---')
        segment(speed_threshold=args.speed_threshold, time_threshold=args.time_threshold)

    if begun or args.mode in ['match', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---###  Matching data  ###---')
        match_data(skip_gopro=args.skip_gopro)
    
    if begun or args.mode in ['resample', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---### Resampling data ###---')
        resample(verbose=args.verbose)
    
    if begun or args.mode in ['kpi', 'all']:
        if not begun and args.begin_from:
            begun = True
        print('    ---### Calculating KPIs ###---')
        compute_kpis()


def get_args(external_parser : ArgumentParser = None) -> Namespace:
    if external_parser is None:
        parser = ArgumentParser()
        parser.add_argument('mode', type=str, choices=['convert', 'validate', 'segment', 'match', 'resample', 'kpi', 'all'], help='Mode to run the script in (all runs all modes in sequence)')
    else:
        parser = external_parser
        
    parser.add_argument('--begin-from', action='store_true', help='Start from specified mode (inclusive)')
    parser.add_argument('--skip-gopro', action='store_true', help='Skip GoPro data in all steps')
    parser.add_argument('--speed-threshold', type=int, default=5, help='Speed threshold for segmenting data (km/h)')
    parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for segmenting data')
    parser.add_argument('--validation-threshold', type=float, default=0.2, help='Normalised MSE threshold for validating data')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    if external_parser is None:
        return parser.parse_args()
    else:
        return parser

if __name__ == '__main__':
    args = get_args()
    main(args)