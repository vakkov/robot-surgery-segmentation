#from prepare_data import data_path
import os
from pathlib import Path

#data_path = os.path.join(os.path.abspath(__file__), 'data')
data_path = Path(__file__).resolve().parent / 'data'

def get_split(fold):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = Path.joinpath(data_path, 'cropped_train')

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names
