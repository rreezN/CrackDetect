from time import time
from argparse import ArgumentParser, Namespace

from src.util.utils import set_all_seeds
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import get_args as get_make_dataset_args
from src.data.feature_extraction import main as feature_extraction
from src.data.feature_extraction import get_args as get_feature_extraction_args
from src.train_hydra_mr import main as train_model
from src.train_hydra_mr import get_args as get_train_model_args
from src.predict_model import main as predict_model
from src.predict_model import get_args as get_predict_model_args


def get_args():
    parser = ArgumentParser(conflict_handler='resolve')
    parser = get_make_dataset_args(parser)
    parser.add_argument('--mode', type=str, default='all', choices=['convert', 'validate', 'segment', 'match', 'resample', 'kpi', 'all'], help='Mode to run the script in (all runs all modes in sequence)')
    parser = get_feature_extraction_args(parser)
    parser = get_train_model_args(parser)
    parser = get_predict_model_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time()
    
    # Get args from each module
    args = get_args()
    
    # Override args
    # NOTE: Outcomment this if you want to be able to modify which features to extract.
    # HOWEVER, we recommend using the feature_extraction.py script if you want greater control over the feature extraction process.
    # Similarly for the training args (train_hydra_mr.py script)
    
    # Feature extraction args
    args.feature_extractor = 'hydra'
    args.all_cols_wo_location = True
        
    # Training args
    args.epochs = 50
    args.lr = 0.001
    args.feature_extractors = ['HydraMV_8_64']
    args.hidden_dim = 256
    args.model_depth = 5
    args.weight_decay = 0.0
    args.dropout = 0.0
    args.batch_norm = True
    args.batch_size = 64
    args.model = 'models/HydraMRRegressor.pt'
    
    set_all_seeds(args.seed)
    
    # Preprocess raw data
    preprocess_start_time = time()
    print('#######################################################')
    print('#                                                     #')
    print('#                 Preprocessing Data                  #')
    print('#                                                     #')
    print('#######################################################')
    
    make_dataset(args)
    
    preprocess_end_time = time() - preprocess_start_time
    print(' ----------------------------------------------------- ')
    print('Finished preprocessing data')
    print('Processed data is stored in data/processed/w_kpis/segments.hdf5')
    print()
    
    # Extract features
    feature_extraction_start_time = time()
    print('#######################################################')
    print('#                                                     #')
    print('#                 Extracting Features                 #')
    print('#                                                     #')
    print('#######################################################')
    
    feature_extraction(args)
    
    feature_extraction_end_time = time() - feature_extraction_start_time
    print(' ----------------------------------------------------- ')
    print('Finished extracting features')
    print('Features are stored in data/processed/features.hdf5')
    
    # Train model
    train_model_start_time = time()
    print('#######################################################')
    print('#                                                     #')
    print('#                 Training Model                      #')
    print('#                                                     #')
    print('#######################################################')
    
    train_model(args)
    
    train_model_end_time = time() - train_model_start_time
    print(' ----------------------------------------------------- ')
    print('Finished training model')
    print('Trained models are stored in models/ and loss curves are stored in reports/figures/model_results/HydraMRRegressor/')
    print()
    
    # Predict model
    predict_model_start_time = time()
    print('#######################################################')
    print('#                                                     #')
    print('#                 Predicting Model                    #')
    print('#                                                     #')
    print('#######################################################')
    
    predict_model(args)
    
    predict_model_end_time = time() - predict_model_start_time
    print(' ----------------------------------------------------- ')
    print('Finished predicting model')
    print('Results are stored in reports/figures/model_results/HydraMRRegressor/')
    print()
    
    # Validate model
    validate_model_start_time = time()
    print('#######################################################')
    print('#                                                     #')
    print('#                 Validating Model                    #')
    print('#                                                     #')
    print('#######################################################')
    
    # validate_model(args)
    
    validate_model_end_time = time() - validate_model_start_time
    print(' ----------------------------------------------------- ')
    print('Finished validating model')
    print()
    
    
    # Print total time taken
    end_time = time() - start_time
    total_minutes = int(end_time / 60)
    total_seconds = int(end_time % 60)
    
    preprocess_minutes = int(preprocess_end_time / 60)
    preprocess_seconds = int(preprocess_end_time % 60)
    
    feature_extraction_minutes = int(feature_extraction_end_time / 60)
    feature_extraction_seconds = int(feature_extraction_end_time % 60)
    
    train_model_minutes = int(train_model_end_time / 60)
    train_model_seconds = int(train_model_end_time % 60)
    
    predict_model_minutes = int(predict_model_end_time / 60)
    predict_model_seconds = int(predict_model_end_time % 60)
    
    validate_model_minutes = int(validate_model_end_time / 60)
    validate_model_seconds = int(validate_model_end_time % 60)
    
    print(' ----------------------------------------------------- ')
    print(f'Finished in {total_minutes} minutes and {total_seconds} seconds')
    print(f'Preprocesssing took {preprocess_minutes} minutes and {preprocess_seconds} seconds')
    print(f'Feature extraction took {feature_extraction_minutes} minutes and {feature_extraction_seconds} seconds')
    print(f'Training model took {train_model_minutes} minutes and {train_model_seconds} seconds')
    print(f'Predicting model took {predict_model_minutes} minutes and {predict_model_seconds} seconds')
    print(f'Validating model took {validate_model_minutes} minutes and {validate_model_seconds} seconds')
    
    