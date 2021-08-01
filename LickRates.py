import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import psytrack as psy
import time
from Pstrack_vocalization_task import *


def general_rate_calc(action_arr, bin_size=500, rate_kind=1):
    lst = action_arr == rate_kind
    lst = lst.astype(np.int16)
    return lst.rolling(int(bin_size), win_type='triang', center=True,
                       min_periods=int(bin_size // 2)).mean()


def lick_rate(df):
    action_arr, answer_arr, correct_arr = get_action_answer_correct(df)
    day_length = get_day_length_by_levels(df)
    levels = np.cumsum(day_length)
    all_rates = []
    for i, current_lvl in enumerate(levels):
        if i == 0:
            current_rate = general_rate_calc(
                pd.Series(action_arr[:current_lvl]))
            all_rates.extend(current_rate.tolist())

        else:
            current_rate = general_rate_calc(
                pd.Series(action_arr[levels[i - 1]:current_lvl]))
            all_rates.extend(current_rate.tolist())

    # plot results
    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(all_rates, zorder=1)
    plt.axhline(y=0.5, color='r', linestyle='--', zorder=2)
    plt.xlabel('Trial #')
    plt.ylabel('Lick Rate %')
    for lvl in levels[:-1]:
        plt.axvline(x=lvl, color='black', linewidth=0.4, zorder=2,
                    linestyle='--')
    plt.ylim([np.min(all_rates)-0.1,1])
    plt.tight_layout()

    plt.show()

    return fig


origin_df = load_data()
cleand_df = clean_data(origin_df)
all_mice_dict = get_all_mice_dict(cleand_df)
for mouse_name in all_mice_dict.keys():
    # if mouse_name != '0007DEC834':
    #     continue
    print(mouse_name)
    mouse_df = all_mice_dict[mouse_name]
    fig_lick_rates = lick_rate(mouse_df)
    #save_fig(fig_lick_rates, mouse_name, 'Lick Rates',dpi=500)
