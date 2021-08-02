import time
from Psytrack_plot_options import *
from Psytrack_data_manager import *
from Psytrack_sessions_handler import *


def get_auditory_stimuli_lst(df):
    stim_arr = df[STIM_ID].values
    cond = [stim_arr < NO_GO_STIM_ID_2, stim_arr < NO_GO_STIM_ID_3,
            stim_arr < NO_GO_STIM_ID_4, stim_arr < NO_GO_STIM_ID_5,
            stim_arr < NO_GO_STIM_ID_6]
    choice = [GO, NO_GO_REV, NO_GO_06, NO_GO_08, NO_GO_09]
    auditory_stimuli_arr = np.select(cond, choice)

    return auditory_stimuli_arr


def get_auditory_stimuli_lst_simple(df):
    stim_arr = df[STIM_ID].values
    auditory_stimuli_arr = np.ones(stim_arr.shape)

    # Hard coded auditory stimuli modeling
    auditory_stimuli_arr[stim_arr == 1] = 1
    auditory_stimuli_arr[stim_arr == 2] = -0.5
    auditory_stimuli_arr[stim_arr == 3] = -0.25
    auditory_stimuli_arr[stim_arr == 4] = -0.125
    auditory_stimuli_arr[stim_arr == 5] = -0.0625

    return auditory_stimuli_arr


def get_go_nogo_weigths(mouse_df):
    stim_arr = mouse_df[STIM_ID].values
    go_weight = np.zeros(stim_arr.shape)
    no_go_weight = np.zeros(stim_arr.shape)
    go_weight[stim_arr == 1] = GO

    # Hard coded go/no go stimuli modeling
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
    cond = [stim_arr < NO_GO_STIM_ID_2, stim_arr >= NO_GO_STIM_ID_2]
    choice = [LICK, NO_LICK]
    answer_arr = np.select(cond, choice)

    # correct:= whether the animal made the correct choice on trail t
    correct_arr = answer_arr == action_arr

    return action_arr, answer_arr, correct_arr


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


def transform_input(weight, p=5, do_transform=False):
    if do_transform:
        return np.tanh(p * weight) / np.tanh(p)
    return weight


def get_psy_dict(mouse_df, analysis_type, history_len):
    auditory_stimuli_arr = get_auditory_stimuli_lst_simple(mouse_df)
    auditory_stimuli_arr = transform_input(auditory_stimuli_arr)

    action_arr, answer_arr, correct_arr = get_action_answer_correct(mouse_df)
    prev_stim, prev_answer, prev_action = [], [], []
    if history_len:
        prev_stim, prev_answer, prev_action = get_all_history_arr(mouse_df,
                                                                  auditory_stimuli_arr,
                                                                  answer_arr,
                                                                  action_arr,
                                                                  history_len)

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


def get_all_mice_data():
    origin_df = load_data()
    cleand_df = clean_data(origin_df)
    all_mice_dict = get_all_mice_dict(cleand_df)
    custom_pre_preprocess(all_mice_dict)
    return all_mice_dict


def run_analysis(history_len=0, analysis_type=GO_NOGO_ANALYSIS):
    """

    :param history_len: 0 is default := no use of history
    """
    all_mice_dict = get_all_mice_data()
    for mouse_name in all_mice_dict.keys():
        print("mouse name: ", mouse_name)
        t1 = time.time()
        mouse_df = all_mice_dict[mouse_name]
        psy_dict = get_psy_dict(mouse_df, analysis_type, history_len)
        weights, K = get_weights(history_len, analysis_type)
        print("preprocess done, waiting for psytrack...")
        for opt_lst in ALL_OPT_TYPES:
            print('optimization for: ', opt_lst)
            res_dict, param_dict = run_optimization(psy_dict, weights, K,
                                                    opt_lst)
            save_dict_data(mouse_name, res_dict, param_dict)
            plot_res(psy_dict, res_dict, param_dict, mouse_name, mouse_df)
            t2 = time.time()
            print("{} done in {}".format(mouse_name, t2 - t1))


if __name__ == '__main__':
    run_analysis(history_len=1, analysis_type=AUDITORY_ANALYSIS)
