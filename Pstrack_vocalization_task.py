import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import psytrack as psy
import time
import pickle
import os
import datetime
import scipy.stats
from sklearn.cluster import KMeans

####################################### DATA PATHS ###########################

IN_DATA_PATH = 'data\\EducageTable.csv'
OUT_DATA_PATH = "data\\{}-processed.csv"
CURRENT_ANALYSIS_PATH = 'C:\\Users\\razco\\PycharmProjects\\MLCage\\data\\current_analysis'

####################################### DATA PATHS ###########################
AUDITORY_STIMULUS = 'auditory stimulus'
AUDITORY_STIMULUS_GO = 'Go stimulus'
AUDITORY_STIMULUS_NO_GO = 'NoGo stimulus'
GO_NOGO_ANALYSIS = "Go/NoGO Analysis"
AUDITORY_ANALYSIS = "Auditory Analysis"

PREV_STIM = 'previous stimulus'
PREV_ANSWER = 'previous answer'
PREV_ACTION = 'previous action'
MOUSE_NAME = "mouse_name"
LEVEL_NUM = "level"
TIME = "time"
LEVEL_NAME = 'levelname'
STIM_ID = "stimID"
SCORE = "score"
GO_WEIGHT = "Go stimulus"
LICK_BIAS = "Lick Bias"
NO_GO_WEIGHT = "NoGo stimulus"
HAB_LEVELS = ['Hab', 'Association', 'AssociationHab']
STIM_TO_REMOVE = [6, 7, 8]
NO_GO_LST = [2, 3, 4, 5]
MIN_NO_GO_STIM_ID = 2
MIN_LICK_SCORE = 2

GO = 1

NO_GO = 0
NO_GO_REV = 0
NO_GO_06 = -0.6
NO_GO_08 = -0.8
NO_GO_09 = -0.9
LICK = 1
NO_LICK = 0

DELTA_MINUTES = 0
DELTA_HOURS = 2
MIN_SESSION_NUM = 10

RES_INDX = 0
PARAM_INDX = 1
RES_DICT = '{}-res_dict'
PARAM_DICT = '{}-param_dict'
PSY_DICT = '{}-psy_dict'

W_BIAS_IND = 1
W_AUDITORY_IND = 0
W_PREV_ACTION_IND = 2
AVG_BIN_SIZE = 20

# ALL_OPT_TYPE_LST = [['sigma', 'sigDay'], ['sigma']]
ALL_OPT_TYPE_LST = [['sigma', 'sigDay']]
# ALL_OPT_TYPE_LST = [['sigma']]

COLORS = {'bias': '#FAA61A', AUDITORY_STIMULUS: "#A9373B",
          PREV_STIM: '#99CC66', PREV_ANSWER: '#9593D9', PREV_ACTION: '#59C3C3',
          GO_WEIGHT: "#A9373B", NO_GO_WEIGHT: "#2369BD", LICK_BIAS: '#FAA61A'}
ZORDER = {'bias': 2, AUDITORY_STIMULUS: 3, PREV_STIM: 3, PREV_ANSWER: 3,
          PREV_ACTION: 3, GO_WEIGHT: 3, NO_GO_WEIGHT: 3, LICK_BIAS: 2}


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


def get_auditory_stimuli_lst(df):
    stim_arr = df[STIM_ID].values
    cond = [stim_arr < MIN_NO_GO_STIM_ID, stim_arr < 3, stim_arr < 4,
            stim_arr < 5, stim_arr < 6]
    choice = [GO, NO_GO_REV, NO_GO_06, NO_GO_08, NO_GO_09]
    auditory_stimuli_arr = np.select(cond, choice)

    return auditory_stimuli_arr


def get_auditory_stimuli_lst_simple(df):
    stim_arr = df[STIM_ID].values

    auditory_stimuli_arr = np.ones(stim_arr.shape)

    # auditory_stimuli_arr[stim_arr == 2] = 0.0625
    # auditory_stimuli_arr[stim_arr == 3] = 0.125
    # auditory_stimuli_arr[stim_arr == 4] = 0.25
    # auditory_stimuli_arr[stim_arr == 5] = 0.5

    auditory_stimuli_arr[stim_arr == 1] = 1
    auditory_stimuli_arr[stim_arr == 2] = -0.5
    auditory_stimuli_arr[stim_arr == 3] = -0.25
    auditory_stimuli_arr[stim_arr == 4] = -0.125
    auditory_stimuli_arr[stim_arr == 5] = -0.0625

    # cond = [stim_arr < MIN_NO_GO_STIM_ID, stim_arr >= MIN_NO_GO_STIM_ID]
    # choice = [GO, -1]
    # auditory_stimuli_arr = np.select(cond, choice)

    return auditory_stimuli_arr


def get_go_nogo_weigths(mouse_df):
    stim_arr = mouse_df[STIM_ID].values
    go_weight = np.zeros(stim_arr.shape)
    no_go_weight = np.zeros(stim_arr.shape)

    # cond = [stim_arr < MIN_NO_GO_STIM_ID, stim_arr >= MIN_NO_GO_STIM_ID]
    # choice = [GO, NO_GO]
    # go_weight = np.select(cond, choice)
    go_weight[stim_arr == 1] = GO

    # no_go_weight[stim_arr == 2] = 0
    # no_go_weight[stim_arr == 3] = -0.6
    # no_go_weight[stim_arr == 4] = -0.8
    # no_go_weight[stim_arr == 5] = -0.9

    no_go_weight[stim_arr == 2] = 0.5
    no_go_weight[stim_arr == 3] = 0.25
    no_go_weight[stim_arr == 4] = 0.125
    no_go_weight[stim_arr == 5] = 0.0625
    return go_weight, no_go_weight


def get_action_answer_correct(df):
    stim_arr = df[STIM_ID].values
    scores_arr = df[SCORE].values

    # action=: lick or not on trail t
    cond = [scores_arr < MIN_LICK_SCORE, scores_arr >= MIN_LICK_SCORE]
    choice = [LICK, NO_LICK]
    action_arr = np.select(cond, choice)

    # answer:= correct action on trail t
    cond = [stim_arr < MIN_NO_GO_STIM_ID, stim_arr >= MIN_NO_GO_STIM_ID]
    choice = [LICK, NO_LICK]
    answer_arr = np.select(cond, choice)

    # correct:= whether the animal made the correct choice on trail t
    correct_arr = answer_arr == action_arr

    return action_arr, answer_arr, correct_arr


def get_day_length_by_levels(mouse_df):
    all_levels = np.sort(np.unique(mouse_df[LEVEL_NUM]))
    day_length = []
    for level in all_levels:
        day_length.append(mouse_df.loc[mouse_df[LEVEL_NUM] == level].shape[0])
    return day_length


def get_time_object(trial_time):
    date_and_time = trial_time.split(" ")
    date = date_and_time[0].split('/')
    year = date[2]
    month = date[1]
    day = date[0]
    time = date_and_time[1].split(":")
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


def reduce_sessions(general_day_length):
    general_day_length_lst = general_day_length.tolist()
    reduced = []
    for i, sessions_num in enumerate(general_day_length_lst):
        if i == len(general_day_length_lst) - 1:
            if sessions_num < MIN_SESSION_NUM:
                reduced[-1] += sessions_num
            break

        if sessions_num > MIN_SESSION_NUM:
            reduced.append(sessions_num)
        else:
            general_day_length_lst[i + 1] += sessions_num
    return reduced


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


def get_history_by_arr(trails_arr, arr, history_len, map_values):
    t = trails_arr

    prior = ((t[history_len:] - t[:-history_len]) == history_len).astype(int)
    pad_arr = [0 for _ in range(history_len)]
    prior = np.hstack((pad_arr, prior))

    if map_values:
        # Calculate previous correct answer
        arr = (arr * 2 - 1).astype(int)  # map from (0,1) to (-1,1)

    prev_arr = arr[:-history_len]
    prev_arr = np.hstack((pad_arr, prev_arr))
    # for trials without a valid previous trial, set to 0
    return prev_arr * prior


def get_all_history_arr(mouse_df, auditory_stimuli_arr, answer_arr, action_arr,
                        history_len):
    # Determine which trials do not have a valid previous trial (mistrial or session boundary)
    trails_arr = np.array(mouse_df.index)

    # previous stimulus
    prev_stim = []
    prev_action = []
    prev_answer = []
    for i in range(1, history_len + 1):
        prev_stim.append(
            get_history_by_arr(trails_arr, auditory_stimuli_arr, i,
                               map_values=False))
        prev_answer.append(
            get_history_by_arr(trails_arr, answer_arr, i, map_values=True))
        prev_action.append(
            get_history_by_arr(trails_arr, action_arr, i, map_values=True))

    return np.vstack(prev_stim).T, np.vstack(prev_answer).T, np.vstack(
        prev_action).T


def transform_input(weight, p=5):
    # return np.tanh(p * weight) / np.tanh(p)
    return weight


def get_psy_dict(mouse_df, analysis_type, history_len):
    # auditory_stimuli_arr = get_auditory_stimuli_lst(mouse_df)
    auditory_stimuli_arr = get_auditory_stimuli_lst_simple(mouse_df)
    auditory_stimuli_arr = transform_input(auditory_stimuli_arr)

    action_arr, answer_arr, correct_arr = get_action_answer_correct(mouse_df)
    prev_stim, prev_answer, prev_action = [], [], []
    if history_len:
        prev_stim, prev_answer, prev_action = get_all_history_arr(mouse_df,
                                                                  auditory_stimuli_arr,
                                                                  answer_arr,
                                                                  action_arr,
                                                                  history_len)  # if history_len == 1:  #     prev_stim = prev_stim.T[..., np.newaxis]  #     prev_answer = prev_answer.T[..., np.newaxis]  #     prev_action = prev_action.T[..., np.newaxis]

    if analysis_type == AUDITORY_ANALYSIS:
        # add dimension to each input array as asked in psytrack doc

        audi_stimului = auditory_stimuli_arr[..., np.newaxis]

        lick_bias = np.ones(action_arr.shape)
        lick_bias = -1 * lick_bias
        lick_bias = lick_bias[..., np.newaxis]

        inputs = {AUDITORY_STIMULUS: audi_stimului, PREV_STIM: prev_stim,
                  PREV_ANSWER: prev_answer, PREV_ACTION: prev_action,
                  LICK_BIAS: lick_bias}

    elif analysis_type == GO_NOGO_ANALYSIS:
        # add dimension to each input array as asked in psytrack doc
        lick_bias = np.zeros(action_arr.shape)
        lick_bias[action_arr == 1] = 0.5
        lick_bias = lick_bias[..., np.newaxis]

        go_weight, no_go_weight = get_go_nogo_weigths(mouse_df)
        # add dimension to each input array as asked in psytrack doc
        go_weight = transform_input(go_weight)
        go_weight = go_weight[..., np.newaxis]

        no_go_weight = transform_input(no_go_weight)
        no_go_weight = no_go_weight[..., np.newaxis]

        inputs = {LICK_BIAS: lick_bias, GO_WEIGHT: go_weight,
                  NO_GO_WEIGHT: no_go_weight, PREV_STIM: prev_stim,
                  PREV_ANSWER: prev_answer, PREV_ACTION: prev_action}

    # data_dict = dict(y=action_arr, inputs=inputs, answer=answer_arr,
    #                  correct=correct_arr,
    #                  dayLength=get_day_length_by_levels(mouse_df))
    data_dict = dict(y=action_arr, inputs=inputs, answer=answer_arr,
                     correct=correct_arr,
                     dayLength=get_day_length_by_levels_and_time(mouse_df),
                     level_ind=get_day_length_by_levels(mouse_df))

    return data_dict


def get_weights(history_len, analysis_type):
    if analysis_type == AUDITORY_ANALYSIS:
        weights = {'bias': 1,  # a special key
                   # use only the first column of Auditory Stimuli from inputs
                   AUDITORY_STIMULUS: 1, PREV_ACTION: history_len,
                   PREV_ANSWER: 0, PREV_STIM: 0, LICK_BIAS: 0}
    elif analysis_type == GO_NOGO_ANALYSIS:
        weights = {'bias': 1,  # a special key
                   # use only the first column of Auditory Stimuli from inputs
                   GO_WEIGHT: 1, NO_GO_WEIGHT: 1, PREV_ACTION: history_len,
                   PREV_ANSWER: history_len, PREV_STIM: 0, LICK_BIAS: 0}

    # It is often useful to have the total number of weights K in your model
    K = np.sum([weights[i] for i in weights.keys()])
    return weights, K


def get_hyper(K, opt_lst):
    if len(opt_lst) == 2:
        # indicate the trials where the sigDay variability should supercede the standard sigma variability.
        sigDay = [2 ** -4] * K
    else:
        # Indicates that session boundaries will be ignored in the optimization
        sigDay = None

    hyper = {'sigInit': 2 ** 4.,
             # Set to a single, large value for all weights. Will not be optimized further.
             'sigma': [2 ** -4.] * K,
             # Each weight will have it's own sigma optimized, but all are initialized the same
             'sigDay': sigDay}
    return hyper


def run_optimization(psy_dict, weights, K, opt_lst):
    hyper = get_hyper(K, opt_lst)
    optList = opt_lst
    hyp, evd, wMode, hess_info = psy.hyperOpt(psy_dict, hyper, weights,
                                              optList)
    res_dict = dict(hyp=hyp, evd=evd, wMode=wMode, hess_info=hess_info)
    param_dict = dict(weights=weights, hyper=hyper, optList=optList)

    return res_dict, param_dict


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

    # drop level 8
    for mouse in ['0007DEC60D', '0007DEC834']:
        df = all_mice_dict[mouse]
        all_mice_dict[mouse] = df[df[LEVEL_NUM] != 7]

    return all_mice_dict


def save_fig(fig, mouse_name, title, dpi=700, path=None):
    if path is None:
        path = './data/figures/{0}-{1}'.format(mouse_name, title)
    else:
        path = path + "/{}".format(mouse_name)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_res(psy_dict, res_dict, param_dict, mouse_name, mouse_df):
    days = psy_dict["dayLength"]
    days = np.array(days)
    days = days.reshape(days.size, )

    # with error bar
    fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'],
                           colors=COLORS, zorder=ZORDER,
                           errorbar=res_dict['hess_info']["W_std"], days=days)

    # fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'], days=days,
    #                        colors=COLORS, zorder=ZORDER)

    all_levels = np.cumsum(get_day_length_by_levels(mouse_df))
    for level in all_levels:
        plt.axvline(level, c='blue', ls='-', lw=0.8, alpha=0.5, zorder=0)

    # plt.legend(loc='upper left', prop={'size': 6})

    save_fig(fig, mouse_name, 'weights')
    # plt.show()

    fig_perf = psy.plot_performance(psy_dict)
    save_fig(fig_perf, mouse_name, 'performance')
    # plt.show()

    fig_bias = psy.plot_bias(psy_dict)
    save_fig(fig_bias, mouse_name, 'bias')  # plt.show()


def save_dict_data(mouse_name, res_dict, param_dict):
    with open('./data//data_dicts//' + RES_DICT.format(mouse_name) + '.pkl',
              'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data//data_dicts//' + PARAM_DICT.format(mouse_name) + '.pkl',
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

    data_dicts_files = os.listdir('./data//data_dicts')
    for fname in data_dicts_files:
        fname_lst = fname.split('-')
        mouse_name, dict_type = fname_lst[0], fname_lst[1].split('.')[0]
        d = {}
        with open('./data//data_dicts//' + fname, 'rb') as handle:
            d = pickle.load(handle)

        if mice_data_dict.get(mouse_name) is None:
            mice_data_dict[mouse_name] = {dict_type: d}
        else:
            mice_data_dict[mouse_name][dict_type] = d

    return mice_data_dict


def run_analysis(history_len=0, analysis_type=GO_NOGO_ANALYSIS):
    """

    :param history_len: 0 is default := no use of history
    :param analysis_type:
    :return:
    """
    origin_df = load_data()
    cleand_df = clean_data(origin_df)
    all_mice_dict = get_all_mice_dict(cleand_df)
    all_mice_dict = custom_pre_preprocess(all_mice_dict)
    for mouse_name in all_mice_dict.keys():

        print("mouse name: ", mouse_name)
        t1 = time.time()
        mouse_df = all_mice_dict[mouse_name]
        psy_dict = get_psy_dict(mouse_df, analysis_type, history_len)

        weights, K = get_weights(history_len, analysis_type)
        print("preprocess done, waiting for psytrack...")
        for opt_lst in ALL_OPT_TYPE_LST:
            print('optimization for: ', opt_lst)
            res_dict, param_dict = run_optimization(psy_dict, weights, K,
                                                    opt_lst)
            save_dict_data(mouse_name, res_dict, param_dict)
            plot_res(psy_dict, res_dict, param_dict, mouse_name, mouse_df)
            t2 = time.time()
            print("{} done in {}".format(mouse_name, t2 - t1))


def figure_dir_2_dict(dir_path=CURRENT_ANALYSIS_PATH):
    mice_figures_dict = {}
    files = os.listdir(dir_path)
    for fname in files:
        image = mpimg.imread(dir_path + '//' + fname)

        fname_lst = fname.split('-')
        mouse_name, plot_type = fname_lst[0], fname_lst[1].split('.')[0]
        if mice_figures_dict.get(mouse_name) is None:
            mice_figures_dict[mouse_name] = {plot_type: image}
        else:
            if mice_figures_dict[mouse_name].get(plot_type) is None:
                mice_figures_dict[mouse_name][plot_type] = image
            else:
                print('2 figures type for same mouse')

    return mice_figures_dict


def one_plot():
    mice_figures_dict = figure_dir_2_dict()
    print(0)
    for mouse in mice_figures_dict.keys():
        figures_dict = mice_figures_dict[mouse]
        columns = 2
        rows = 2
        dpi = 20

        im_data = figures_dict['weights']
        height, width, depth = im_data.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)

        fig.add_subplot(rows, columns, 1)
        plt.imshow(figures_dict['weights'], aspect="auto",
                   interpolation='none')
        plt.axis('off')

        fig.add_subplot(rows, columns, 2)
        plt.imshow(figures_dict['Lick Rates'], aspect="auto",
                   interpolation='none')
        plt.axis('off')

        fig.add_subplot(rows, columns, 3)
        plt.imshow(figures_dict['bias'], aspect="auto", interpolation='none')
        plt.axis('off')

        fig.add_subplot(rows, columns, 4)
        plt.imshow(figures_dict['performance'], aspect="auto",
                   interpolation='none')
        plt.axis('off')
        title = "mouse id: ".format(mouse)
        plt.title(title)

        # plt.show()
        out_dir_path = "./data/one_plot/{}".format(mouse)

        plt.savefig(out_dir_path, bbox_inches='tight', dpi=100)
        print(mouse)


def _get_axis_boundary_weights_ratio(arr):
    max_y = np.max(arr[W_AUDITORY_IND])
    min_y = np.min(arr[W_AUDITORY_IND])

    max_x = np.max(arr[W_BIAS_IND])
    min_x = np.min(arr[W_BIAS_IND])
    boundary = [min_x, max_x, min_y, max_y]
    for i, point in enumerate(boundary):
        if point > 0:
            boundary[i] += 1
        else:
            boundary[i] -= 1
    return boundary


def _get_axis_boundary_aligned_weights(arr, all_lvls_lst):
    max_y = np.max(arr)
    min_y = np.min(arr)

    all_len_lst = []
    for lvl in all_lvls_lst:
        all_len_lst.append(np.max([len(l) for l in lvl]))

    max_x = np.max(all_len_lst)

    boundary = [0, max_x, min_y, max_y]
    for i, point in enumerate(boundary):
        if not point:
            continue
        if point > 0:
            boundary[i] += 1
        else:
            boundary[i] -= 1
    return boundary


def _data2avg_bins(x, y):
    avg_bins_lst = []
    for data in [x, y]:
        equal_split_size = len(data) // AVG_BIN_SIZE
        split_lst = np.split(data[:equal_split_size * AVG_BIN_SIZE],
                             equal_split_size)

        avg_lst = np.average(np.vstack(split_lst), axis=1)
        # avg_lst = np.append(avg_lst,
        #                     np.average(data[equal_split_size * AVG_BIN_SIZE:]))

        avg_bins_lst.append(avg_lst.tolist())

    return np.array(avg_bins_lst[0]), np.array(avg_bins_lst[1])

def find_clusters(x,y,num_lvls):

    X=[]
    for i in range(len(x)):
       X.append([x[i],y[i]])

    kmeans = KMeans(n_clusters=num_lvls, random_state=0).fit(X)
    predicted_labels=kmeans.labels_

    color_lst =[]
    label2color_dict={0:'blue',1:'c',2:'green',3:'red',4:'m'}
    for i in range(len(x)):
        color_lst.append(label2color_dict[predicted_labels[i]])

    plt.scatter(x,y,c=color_lst)
    plt.show()


def plot_weights_ratio(plot_all_data_points=False,run_clustering=True):
    data_dicts = load_dict_data()
    for mouse in data_dicts.keys():
        print(mouse)
        if mouse != '0007DEC60D':
            continue

        weights_arr = data_dicts[mouse]['res_dict']['wMode']
        levels_ind = np.cumsum(data_dicts[mouse]['psy_dict']['level_ind'])

        # run clustering
        if run_clustering:
            find_clusters(weights_arr[W_BIAS_IND],weights_arr[W_AUDITORY_IND],len(levels_ind))

        if not plot_all_data_points:
            fig, ax = plt.subplots(nrows=1, ncols=len(levels_ind))

        for i in range(len(levels_ind)):
            if not i:
                start_ind = 0
                end_ind = levels_ind[i]
            else:
                start_ind = levels_ind[i - 1]
                end_ind = levels_ind[i]

            y = weights_arr[W_AUDITORY_IND][start_ind:end_ind]
            x = weights_arr[W_BIAS_IND][start_ind:end_ind]
            if plot_all_data_points:
                plt.scatter(x, y, label=i + 1, alpha=0.2)
                plt.axhline(y=0, color='k')
                plt.axvline(x=0, color='k')
            else:
                x, y = _data2avg_bins(x, y)
                print('num data points: {} for level {}'.format(len(x), i + 1))
                boundary = _get_axis_boundary_weights_ratio(weights_arr)
                slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

                sub_plot_title = r"$\bf{" + "Level: " + str(
                    i + 1) + "}$" + "\npvalue={1}\nstderr={2}\nrvalue={3}".format(
                    i + 1, "%.2f" % p, "%.2f" % r, "%.2f" % stderr)

                current_fig = fig.add_subplot(1, len(levels_ind), i + 1)
                ax[i].set_title(sub_plot_title, fontsize=10)

                ax[i].axis('off')
                plt.scatter(x, y, linewidth=0, marker='s', c='r')
                plt.plot(x, intercept + slope * x)

                plt.axhline(y=0, color='k')
                plt.axvline(x=0, color='k')

                plt.xlim([boundary[0], boundary[1]])
                plt.ylim([boundary[2], boundary[3]])

        if plot_all_data_points:
            plt.legend()
            plt.show()
        else:
            fig.text(0.5, 0.04, s='Bias Weight', ha='center', fontsize=10)
            fig.text(0.04, 0.5, 'Auditory Weight', va='center',
                     rotation='vertical', fontsize=10)
            fig.text(0, 0.95, "Mouse: {}".format(mouse), va='center',
                     fontsize=12)

            # save_fig(fig,mouse,"weights_correlations")
            plt.show()


def _alinged_weight_avg(cur_lvl_lst):
    cur_lvl_lst_copy = cur_lvl_lst.copy()
    max_iter = np.ceil(0.2 * len(cur_lvl_lst_copy))
    start_idx = 0
    avg_lst = []
    # continue until reach 4/5 from origin len
    while (len(cur_lvl_lst_copy) > max_iter):
        len_lst = [len(l) for l in cur_lvl_lst_copy]
        lst_idx_to_del = np.argmin(len_lst)
        min_len = np.min(len_lst)

        # avg to this point
        if min_len - start_idx <= 0:
            start_idx = min_len
            del cur_lvl_lst_copy[lst_idx_to_del]
            continue
        mat = np.vstack(
            [l[start_idx:start_idx + (min_len - start_idx)] for l in
             cur_lvl_lst_copy])
        start_idx = min_len

        # concat avg
        avg_lst.extend(np.average(mat, axis=0).tolist())
        del cur_lvl_lst_copy[lst_idx_to_del]
    return avg_lst


def aligned_weight_per_level():
    data_dicts = load_dict_data()
    for mouse in data_dicts.keys():
        print(mouse)
        # and mouse !='0007DEC60D':
        if mouse != '0007D2D1B0':
            continue

        weights_arr = data_dicts[mouse]['res_dict']['wMode']
        inner_lvl_start_ind = np.cumsum(
            data_dicts[mouse]['psy_dict']['dayLength'])
        lvl_start_ind = np.cumsum(
            data_dicts[mouse]['psy_dict']['level_ind'])

        lvl_auditory_lst = []
        lvl_bias_lst = []
        for lvl in lvl_start_ind:
            cur_lvl_auditory_lst = []
            cur_lvl_bias_lst = []
            for i, inner_lvl in enumerate(inner_lvl_start_ind):
                if inner_lvl > lvl:
                    break
                if not i:
                    start_ind = 0
                else:
                    start_ind = inner_lvl_start_ind[i - 1]
                cur_lvl_auditory_lst.append(
                    weights_arr[W_AUDITORY_IND][start_ind:inner_lvl])
                cur_lvl_bias_lst.append(
                    weights_arr[W_BIAS_IND][start_ind:inner_lvl])

            lvl_auditory_lst.append(cur_lvl_auditory_lst)
            lvl_bias_lst.append(cur_lvl_bias_lst)

        for i, w in enumerate([lvl_auditory_lst, lvl_bias_lst]):
            fig, ax = plt.subplots(nrows=1, ncols=len(lvl_auditory_lst))
            fig.text(0.5, 0.04, s='Trial', ha='center', fontsize=10)
            fig.text(0, 0.95, "Mouse: {}".format(mouse), va='center',
                     fontsize=12)
            if i == 0:
                fig.text(0.04, 0.5, 'Auditory Weight', va='center',
                         rotation='vertical', fontsize=10)
                boundary = _get_axis_boundary_aligned_weights(
                    weights_arr[W_AUDITORY_IND], lvl_auditory_lst)
            else:
                fig.text(0.04, 0.5, 'Bias Weight', va='center',
                         rotation='vertical', fontsize=10)
                boundary = _get_axis_boundary_aligned_weights(
                    weights_arr[W_BIAS_IND], lvl_auditory_lst)
            for j, lvl in enumerate(w):
                # calc avg for cur lvl
                avg_arr = _alinged_weight_avg(lvl)
                sub_plot_title = r"$\bf{" + "Level: " + str(j + 1) + "}$"

                current_fig = fig.add_subplot(1, len(lvl_auditory_lst),
                                              j + 1)
                plt.axhline(y=0, color='k')
                plt.axvline(x=0, color='k')
                ax[j].set_title(sub_plot_title, fontsize=10)
                ax[j].axis('off')
                for cur_w in lvl:
                    if not i:
                        # bias plot
                        plt.plot(cur_w, c='#FAA61A', alpha=0.4, linewidth=3)
                        plt.plot(avg_arr, c='black',
                                 linestyle='--')
                    else:
                        # auditory plot
                        plt.plot(cur_w, c='#A9373B', alpha=0.4, linewidth=3)
                        plt.plot(avg_arr, c='black',
                                 linestyle='--')
                    plt.xlim([boundary[0], boundary[1]])
                    plt.ylim([boundary[2], boundary[3]])

            plt.show()


###################################### *Run Commands *#########################
# run_analysis(use_history=False,
#              analysis_type=GO_NOGO_ANALYSIS)
# AUDITORY_ANALYSIS
# run_analysis(history_len=1, analysis_type=AUDITORY_ANALYSIS)  # one_plot()


plot_weights_ratio(plot_all_data_points=True)
#aligned_weight_per_level()
