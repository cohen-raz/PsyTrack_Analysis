import pandas as pd
import numpy as np
import pickle
import os
from Psytrack_constants import *


def load_data(path=IN_DATA_PATH):
    df = pd.read_csv(path)
    return df


def save_data(df, mouse_name):
    df.to_csv(OUT_DATA_PATH.format(mouse_name), header=False)
    print("Data Saved Successfully")


def clean_data(df):
    # drop habituation levels
    df = df[~df[LEVEL_NAME].isin(HAB_LEVELS)]
    # drop unnecessary STIM_ID levels
    df = df[~df[STIM_ID].isin(STIM_TO_REMOVE)]
    return df


def get_all_mice_dict(df):
    all_mice = np.unique(df[MOUSE_NAME])
    all_mice_dict = {}
    for mouse in all_mice:
        all_mice_dict[mouse] = df.loc[df[MOUSE_NAME] == mouse]
    return all_mice_dict


def save_dict_data(mouse_name, res_dict, param_dict):
    with open(DATA_DICTS_PATH + RES_DICT.format(mouse_name) + PICKL_EXTENSION,
              'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
            DATA_DICTS_PATH + PARAM_DICT.format(mouse_name) + PICKL_EXTENSION,
            'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict_data():
    mice_data_dict = {}
    psy_dict_files = os.listdir('./data//psy_dicts')
    for fname in psy_dict_files:
        fname_lst = fname.split('-')
        mouse_name, dict_type = fname_lst[0], fname_lst[1].split('.')[0]
        d = {}
        with open('./data//psy_dicts//' + fname, 'rb') as handle:
            d = pickle.load(handle)

        if mice_data_dict.get(mouse_name) is None:
            mice_data_dict[mouse_name] = {dict_type: d}
        else:
            mice_data_dict[mouse_name][dict_type] = d

    data_dicts_files = os.listdir(DATA_DICTS_PATH)
    for fname in data_dicts_files:
        fname_lst = fname.split('-')
        mouse_name, dict_type = fname_lst[0], fname_lst[1].split('.')[0]
        d = {}
        with open(DATA_DICTS_PATH + fname, 'rb') as handle:
            d = pickle.load(handle)

        if mice_data_dict.get(mouse_name) is None:
            mice_data_dict[mouse_name] = {dict_type: d}
        else:
            mice_data_dict[mouse_name][dict_type] = d

    return mice_data_dict
