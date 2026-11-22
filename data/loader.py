import numpy as np
import pandas as pd
import os


def data_name2data(data_path: str, data_name: str):

    if data_name == 'MSL' or data_name == 'NIPS_TS_Swan' or data_name == 'SMD':
        root_path = os.path.join(data_path, data_name)
        train_x = np.load(os.path.join(root_path, data_name + '_train.npy'))
        test_x = np.load(os.path.join(root_path, data_name + '_test.npy'))
        test_y = np.load(os.path.join(root_path, data_name + '_test_label.npy')).astype(int)

    elif data_name == 'PSM':
        root_path = os.path.join(data_path, data_name)
        train_x = pd.read_csv(os.path.join(root_path, data_name + '_train.csv'), index_col=0).values.astype(float)
        train_x = np.nan_to_num(train_x)
        test_x = pd.read_csv(os.path.join(root_path, data_name + '_test.csv'), index_col=0).values.astype(float)
        test_x = np.nan_to_num(test_x)
        test_y = pd.read_csv(os.path.join(root_path, data_name + '_test_label.csv'), index_col=0).values.astype(int)

    else:
        raise ValueError(f'no this dataset {data_name}')

    return train_x, test_x, test_y
