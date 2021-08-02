import numpy as np
import datetime
import pandas as pd
from Psytrack_constants import *


def get_day_length_by_levels_and_time(mouse_df):
    all_trials_time = mouse_df[TIME].to_numpy()
    sessions_boundaries = []

    day_length_levels = get_day_length_by_levels(mouse_df)
    switch_levels_indx = np.cumsum(day_length_levels)
    last_switch_index = 0
    for i, switch_index in enumerate(switch_levels_indx):
        time_delta_sessions = get_time_delta_sessions(
            all_trials_time[last_switch_index:switch_index])
        sessions_boundaries.extend(
            np.array(time_delta_sessions) + last_switch_index)
        sessions_boundaries.append(day_length_levels[i] + last_switch_index)
        last_switch_index = switch_index
    print("num sessions: ", len(sessions_boundaries))
    general_day_length = np.diff(sessions_boundaries)
    return reduce_sessions(general_day_length)


def reduce_sessions(general_day_length):
    general_day_lengths = general_day_length.tolist()
    reduced = []
    for i, sessions_num in enumerate(general_day_lengths):
        if i == len(general_day_lengths) - 1:
            if sessions_num < MIN_SESSION_NUM:
                reduced[-1] += sessions_num
            break

        if sessions_num > MIN_SESSION_NUM:
            reduced.append(sessions_num)
        else:
            general_day_lengths[i + 1] += sessions_num
    return reduced


def get_day_length_by_levels(mouse_df):
    all_levels = np.sort(np.unique(mouse_df[LEVEL_NUM]))
    day_length = []
    for level in all_levels:
        day_length.append(mouse_df.loc[mouse_df[LEVEL_NUM] == level].shape[0])
    return day_length


def get_time_object(trial_time):
    date_and_time = trial_time.split(" ")
    date = date_and_time[1].split('/')
    year = date[2]
    month = date[1]
    day = date[0]
    time = date_and_time[0].split(":")
    return datetime.datetime(year=int(year), month=int(month), day=int(day),
                             hour=int(time[0]), minute=int(time[1]))


def check_delta(time_a, time_b):
    time_a = get_time_object(time_a)
    time_b = get_time_object(time_b)
    cond = (time_a + datetime.timedelta(minutes=DELTA_MINUTES,
                                        hours=DELTA_HOURS))
    cond = cond < time_b
    return cond


def get_time_delta_sessions(time_arr):
    time_delta_sessions = []
    for i, time_a in enumerate(time_arr):
        if i == len(time_arr) - 1:
            break
        time_b = time_arr[i + 1]
        if check_delta(time_a, time_b):
            time_delta_sessions.append(i + 1)

    return time_delta_sessions


def _merge_levels(df, level_a_num, level_b_num):
    level_a_name = df.loc[df[LEVEL_NUM] == level_a_num, LEVEL_NAME].tolist()[0]
    level_b_name = df.loc[df[LEVEL_NUM] == level_b_num, LEVEL_NAME].tolist()[0]
    merged_name = level_a_name + '+' + level_b_name
    df.loc[df[LEVEL_NUM].isin(
        [level_a_num, level_b_num]), LEVEL_NAME] = merged_name

    levels = pd.unique(df[LEVEL_NUM]).tolist()
    for l in levels:
        if l < level_b_num:
            continue
        df.loc[df[LEVEL_NUM] == l, LEVEL_NUM] = l - 1

    return df


def custom_pre_preprocess(all_mice_dict):
    # merge levels 4-5
    for mouse in ['0007D2CD8B', '0007D2D1B0', '0007D2E150', '0007D31D1A',
                  '0007DEFC4A', '0007E0DDA0']:
        all_mice_dict[mouse] = _merge_levels(all_mice_dict[mouse], 4, 5)

    # merge levels 5-6
    for mouse in ['0007DEC60D', '0007DEC834', '0007DEFB61']:
        all_mice_dict[mouse] = _merge_levels(all_mice_dict[mouse], 5, 6)

    # drop level 7
    for mouse in ['0007DEC60D', '0007DEC834', '0007DEFB61', '0007DEFC4A',
                  '0007E0DDA0']:
        df = all_mice_dict[mouse]
        all_mice_dict[mouse] = df[df[LEVEL_NUM] != 7]

    # drop level 6
    for mouse in ['0007DEFC4A', '0007E0DDA0', '0007DEC60D', '0007DEFB61',
                  '0007DEC834']:
        df = all_mice_dict[mouse]
        all_mice_dict[mouse] = df[df[LEVEL_NUM] != 6]
