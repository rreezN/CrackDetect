from argparse import ArgumentParser, Namespace

from data_functions.converting import convert
from data_functions.extract_gopro import preprocess_gopro_data
from data_functions.validating import validate
from data_functions.segmenting import segment
from data_functions.matching import match_data
from data_functions.resampling import resample
from data_functions.kpis import compute_kpis


def main(args: Namespace) -> None:

    begin_from = False

    if begin_from or args.mode in ['convert', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---### Converting data ###---')
        convert()
        # Convert GoPro data to align with the GM trips
        preprocess_gopro_data()
    
    if begin_from or args.mode in ['validate', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---### Validating data ###---')
        validate(threshold=args.validation_threshold, verbose=args.verbose)

    if begin_from or args.mode in ['segment', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---### Segmenting data ###---')
        segment(speed_threshold=args.speed_threshold, time_threshold=args.time_threshold)

    if begin_from or args.mode in ['match', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---###  Matching data  ###---')
        match_data()
    
    if begin_from or args.mode in ['resample', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---### Resampling data ###---')
        resample(verbose=args.verbose)
    
    if begin_from or args.mode in ['kpi', 'all']:
        if not begin_from and args.begin_from:
            begin_from = True
        print('    ---### Calculating KPIs ###---')
        compute_kpis()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=['convert', 'validate', 'segment', 'match', 'resample', 'kpi', 'all'], help='Mode to run the script in (all runs all modes in sequence)')
    parser.add_argument('--begin-from', action='store_true', help='Start from specified mode (inclusive)')
    parser.add_argument('--speed-threshold', type=int, default=5, help='Speed threshold for segmenting data (km/h)')
    parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for segmenting data')
    parser.add_argument('--validation-threshold', type=float, default=0.2, help='Normalised MSE threshold for validating data')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    main(args)