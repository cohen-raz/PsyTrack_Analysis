import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import psytrack as psy

####################################### DATA PATHS ###########################

IN_DATA_PATH = 'data\\all_mice_all_sessions.xlsx'
OUT_DATA_PATH = "data\\{}-processed.csv"

####################################### DATA PATHS ###########################

OLFACTORY_STIMULUS = 'olfactory stimulus'
AUDITORY_STIMULUS = 'auditory stimulus'
GO_WEIGHT = 'Go weight'
NO_GO_WEIGHT = 'NoGo weight'

AUDITORY_STIMULUS_GO = 'a_stim_go'
AUDITORY_STIMULUS_NO_GO = 'a_stim_no_go'
OLFACTORY_STIMULUS_NETURAL = 'o_stim_natural'
OLFACTORY_STIMULUS_GO = 'o_stim_go'
OLFACTORY_STIMULUS_NO_GO = 'o_stim_no_go'

OLFACTORY_AUDITORY_ANALYSIS = 'weights by stimuli'
GO_NO_GO_ANALYSIS = 'weights by trial type'

# ANALYSIS_TYPE_LST = [OLFACTORY_AUDITORY_ANALYSIS, GO_NO_GO_ANALYSIS]
ANALYSIS_TYPE_LST = [OLFACTORY_AUDITORY_ANALYSIS]
ALL_OPT_TYPE_LST = [['sigma', 'sigDay'], ['sigma']]

COLORS = {'bias': '#FAA61A', AUDITORY_STIMULUS: "#A9373B",
          GO_WEIGHT: "#A9373B", OLFACTORY_STIMULUS: '#2369BD',
          NO_GO_WEIGHT: '#2369BD'}
ZORDER = {'bias': 2, AUDITORY_STIMULUS: 3, GO_WEIGHT: 3, OLFACTORY_STIMULUS: 3,
          NO_GO_WEIGHT: 3}

INIT_ROW_NAMES = [OLFACTORY_STIMULUS_GO, OLFACTORY_STIMULUS_NETURAL,
                  OLFACTORY_STIMULUS_NO_GO, AUDITORY_STIMULUS_NO_GO,
                  AUDITORY_STIMULUS_GO, 'Action', 'Answer', 'Correct']

GO = 1
NO_GO = -1
LICK = 1
NO_LICK = 0
MAX_GO = 3
STIMULI_INT_DICT = {1: 'GO', 0: 'NO_GO', 'GO': 1, 'NO_GO': 0}

GO_STIMULI = [1, 2]
NO_GO_STIMULI = [3, 4]
SCORE_2_ACTION = {1: LICK, 2: NO_LICK, 3: NO_LICK, 4: LICK, 7: -1}
STIMULI_2_ANSWER = {GO: LICK, NO_GO: NO_LICK}

OLFACTORY_DF_INDEX = 1
DAY_LEN_DF_INDEX = 2


def get_row_indx(size):
    return np.arange(0, size, 4).tolist(), np.arange(1, size, 4).tolist()


def all_sessions_df_2_mouse_df(df):
    score_indx_lst, olfactory_indx_arr_lst = get_row_indx(df.shape[0])
    all_score_lst = []
    all_olfactory_lst = []
    dayLength = []
    for index, row in df.iterrows():
        if index in score_indx_lst:
            current_row = row.dropna().values
            dayLength.append(np.count_nonzero(current_row != 7))
            all_score_lst.extend(current_row.tolist())
        elif index in olfactory_indx_arr_lst:
            all_olfactory_lst.extend(row.dropna().values.tolist())

    df = pd.DataFrame(data=[all_score_lst, all_olfactory_lst])
    # remove cols where score == 7 (not from dayLength row)
    df = df.loc[:, ~(df == 7).any()]
    df.columns = range(df.shape[1])

    df_days = pd.DataFrame(data=[dayLength])
    df = df.append(df_days)
    return df


def load_data(path=IN_DATA_PATH):
    # split all mice data to a dict (key=mouse name, value= mouse df)
    all_mice_df_dict = pd.read_excel(io=path, header=None, sheet_name=None)
    for key in all_mice_df_dict.keys():
        all_mice_df_dict[key] = all_sessions_df_2_mouse_df(
            all_mice_df_dict[key])

    return all_mice_df_dict


def save_data(df, mouse_name, path=OUT_DATA_PATH):
    df.to_csv(path.format(mouse_name), header=False)


def get_auditory_stimuli_lst(df, score_idx=0):
    scores = df.iloc[[score_idx], :].values

    cond = [scores < MAX_GO, scores >= MAX_GO]
    choice = [GO, NO_GO]
    auditory_stimuli_arr = np.select(cond, choice)
    return auditory_stimuli_arr, scores


def get_action_answer_correct(scores_lst, auditory_stimuli_arr):
    action_lst = []
    answer_lst = []
    correct_lst = []
    for i, score in enumerate(scores_lst):
        action = SCORE_2_ACTION[score]
        action_lst.append(action)
        answer_lst.append(STIMULI_2_ANSWER[auditory_stimuli_arr[0, i]])

        if action == answer_lst[i]:
            correct_lst.append(1)
        elif action == -1:
            correct_lst.append(-1)
        else:
            correct_lst.append(0)
    return action_lst, answer_lst, correct_lst


def split_stimului_lst(stimuli_lst):
    unique_stimuli_arr = np.unique(np.array(stimuli_lst))
    num_stimuli = unique_stimuli_arr.shape[0]

    all_stim_mat = np.zeros((len(stimuli_lst), num_stimuli))
    for i, stimulus in enumerate(stimuli_lst):
        col_idx = np.where(unique_stimuli_arr == stimulus)[0][0]
        all_stim_mat[i, col_idx] = 1
    return all_stim_mat


def get_olfactiry_stim_arr(df):
    olfactory_stimuli_arr = df.iloc[[OLFACTORY_DF_INDEX], :].values
    # change olfactory stimuli: Go=1, Natural=0, NoGO=-1
    return 2 - olfactory_stimuli_arr  # return olfactory_stimuli_arr


def stim2weight(olfactory_stimuli_arr, auditory_stimuli_arr):
    stim_size = olfactory_stimuli_arr.size
    olfactory_stimuli_arr = olfactory_stimuli_arr.reshape(stim_size, )
    auditory_stimuli_arr = auditory_stimuli_arr.reshape(stim_size, )

    w_go = np.zeros(stim_size)
    w_go[np.logical_and(olfactory_stimuli_arr == 1,
                        auditory_stimuli_arr == 1)] = 0.66
    w_go[np.logical_and(olfactory_stimuli_arr == 0,
                        auditory_stimuli_arr == 1)] = 0.5
    w_go[np.logical_and(olfactory_stimuli_arr == -1,
                        auditory_stimuli_arr == 1)] = 0.33

    w_no_go = np.zeros(stim_size)
    w_no_go[np.logical_and(olfactory_stimuli_arr == 1,
                           auditory_stimuli_arr == -1)] = 0.33
    w_no_go[np.logical_and(olfactory_stimuli_arr == 0,
                           auditory_stimuli_arr == -1)] = 0.5
    w_no_go[np.logical_and(olfactory_stimuli_arr == -1,
                           auditory_stimuli_arr == -1)] = 0.66

    return w_go, w_no_go


def stimuli_pre_processing(df):
    olfactory_stimuli_arr = get_olfactiry_stim_arr(df)
    auditory_stimuli_arr, scores_arr = get_auditory_stimuli_lst(df)

    # action, answer and correct to the new data frame
    action_lst, answer_lst, correct_lst = get_action_answer_correct(
        scores_arr.tolist()[0], auditory_stimuli_arr)

    return pd.DataFrame(
        data={OLFACTORY_STIMULUS: olfactory_stimuli_arr.tolist()[0],
              AUDITORY_STIMULUS: auditory_stimuli_arr.tolist()[0],
              'Action': action_lst, 'Answer': answer_lst,
              'Correct': correct_lst})


def choice_pre_processing(df):
    olfactory_stimuli_arr = get_olfactiry_stim_arr(df)
    auditory_stimuli_arr, scores_arr = get_auditory_stimuli_lst(df)
    # action, answer and correct to the new data frame
    action_lst, answer_lst, correct_lst = get_action_answer_correct(
        scores_arr.tolist()[0], auditory_stimuli_arr)

    w_go, w_no_go = stim2weight(olfactory_stimuli_arr, auditory_stimuli_arr)

    return pd.DataFrame(
        data={GO_WEIGHT: w_go.tolist(), NO_GO_WEIGHT: w_no_go.tolist(),
              'Action': action_lst, 'Answer': answer_lst,
              'Correct': correct_lst})


def df_pre_processing(df, analysis_type):
    if analysis_type == OLFACTORY_AUDITORY_ANALYSIS:
        psy_df = stimuli_pre_processing(df)

    elif analysis_type == GO_NO_GO_ANALYSIS:
        psy_df = choice_pre_processing(df)

    return psy_df.T


def df_2_psy_dict(df, analysis_type, day_length):
    answer = df.loc['Answer'].values
    correct = df.loc['Correct'].values
    # add dimension to each input array as asked in psytrac doc
    if analysis_type == OLFACTORY_AUDITORY_ANALYSIS:
        audi_stimului = df.loc[AUDITORY_STIMULUS].values
        audi_stimului = audi_stimului[..., np.newaxis]

        olfa_stimului = df.loc[OLFACTORY_STIMULUS].values
        olfa_stimului = olfa_stimului[..., np.newaxis]

        inputs = {AUDITORY_STIMULUS: audi_stimului,
                  OLFACTORY_STIMULUS: olfa_stimului}

    elif analysis_type == GO_NO_GO_ANALYSIS:
        go_stim = df.loc[GO_WEIGHT].values
        go_stim = go_stim[..., np.newaxis]

        nogo_stim = df.loc[NO_GO_WEIGHT].values
        nogo_stim = nogo_stim[..., np.newaxis]

        inputs = {GO_WEIGHT: go_stim, NO_GO_WEIGHT: nogo_stim}

    data_dict = dict(y=df.loc['Action'].values, inputs=inputs, answer=answer,
                     correct=correct, dayLength=day_length)

    return data_dict


def get_day_length_arr(df):
    return df.iloc[[DAY_LEN_DF_INDEX], :].dropna(1).values.astype(int)


def pre_processing(df, analysis_type):
    psy_df = df_pre_processing(df, analysis_type)
    psy_dict = df_2_psy_dict(psy_df, analysis_type, get_day_length_arr(df))
    return psy_df, psy_dict


def get_weights(analysis_type):
    if analysis_type == OLFACTORY_AUDITORY_ANALYSIS:
        # weights = {'bias': 1,  # a special key
        #            # use only the first column of Auditory Stimuli from inputs
        #            AUDITORY_STIMULUS: 1, OLFACTORY_STIMULUS: 1}
        weights = {
        # use only the first column of Auditory Stimuli from inputs
            AUDITORY_STIMULUS: 1, OLFACTORY_STIMULUS: 1}

    elif analysis_type == GO_NO_GO_ANALYSIS:
        weights = {'bias': 1,  # a special key
                   # use only the first column of Auditory Stimuli from inputs
                   GO_WEIGHT: 1, NO_GO_WEIGHT: 1}

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


def plot_res(psy_dict, res_dict, param_dict):
    # with error bar
    # fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'],
    #                        colors=COLORS, zorder=ZORDER,
    #                        errorbar=res_dict['hess_info']["W_std"])

    days = psy_dict["dayLength"]
    days = np.array(days)
    days = days.reshape(days.size, )

    fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'], days=days,
                           colors=COLORS, zorder=ZORDER)

    plt.legend()
    plt.show()

    fig_perf = psy.plot_performance(psy_dict)
    plt.show()

    # fig_bias = psy.plot_bias(psy_dict)
    plt.show()


def run_analysis():
    # data handling - loading,processing and saving for each mouse and
    # for each analysis type
    all_mice_df_dict = load_data()
    for mouse in all_mice_df_dict.keys():
        print("mouse name: ", mouse)
        original_df = all_mice_df_dict[mouse]

        for analysis_type in ANALYSIS_TYPE_LST:
            print("analysis type:", analysis_type)
            processed_df, psy_dict = pre_processing(original_df, analysis_type)
            save_data(processed_df, mouse)

            # run psytrack model and plot results

            weights, K = get_weights(analysis_type)
            print("preprocess done, waiting for psytrack...")
            for opt_lst in ALL_OPT_TYPE_LST:
                print('optimization for: ', opt_lst)
                res_dict, param_dict = run_optimization(psy_dict, weights, K,
                                                        opt_lst)
                np.savetxt("{}.csv".format(mouse), res_dict['wMode'],
                           delimiter=",")
                #plot_res(psy_dict, res_dict, param_dict)
        print("---------------------------")


run_analysis()
