from Pstrack_vocalization_task import *
from sklearn.cluster import KMeans
from scipy import stats

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


def _get_axis_boundary_aligned_weights(arr, all_lvls):
    max_y = np.max(arr)
    min_y = np.min(arr)

    all_len = []
    for lvl in all_lvls:
        all_len.append(np.max([len(l) for l in lvl]))

    max_x = np.max(all_len)

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
    avg_bins = []
    for data in [x, y]:
        equal_split_size = len(data) // AVG_BIN_SIZE
        split_lst = np.split(data[:equal_split_size * AVG_BIN_SIZE],
                             equal_split_size)

        avg_lst = np.average(np.vstack(split_lst), axis=1)

        avg_bins.append(avg_lst.tolist())

    return np.array(avg_bins[0]), np.array(avg_bins[1])


def find_clusters(x, y, num_lvls):
    X = []
    for i in range(len(x)):
        X.append([x[i], y[i]])

    kmeans = KMeans(n_clusters=num_lvls, random_state=0).fit(X)
    predicted_labels = kmeans.labels_

    all_colors = []
    label2color_dict = {0: 'blue', 1: 'c', 2: 'green', 3: 'red', 4: 'm'}
    for i in range(len(x)):
        all_colors.append(label2color_dict[predicted_labels[i]])

    plt.scatter(x, y, c=all_colors)
    plt.show()


def _get_requested_data_index(data_len, percentage):
    return int(np.ceil(percentage * data_len))


def plot_weights_ratio(plot_all_data_points=False, run_clustering=True):
    data_dicts = load_dict_data()
    for mouse in data_dicts.keys():
        print(mouse)
        if mouse != '0007DEFC4A':
            continue

        weights_arr = data_dicts[mouse]['res_dict']['wMode']
        levels_ind = np.cumsum(data_dicts[mouse]['psy_dict']['level_ind'])

        # run clustering
        if run_clustering:
            find_clusters(weights_arr[W_BIAS_IND], weights_arr[W_AUDITORY_IND],
                          len(levels_ind))
        drop_index = 0
        if not plot_all_data_points:
            if  weights_arr[W_BIAS_IND].size < levels_ind[-1]:
                if  weights_arr[W_BIAS_IND].size < levels_ind[-2]:
                    drop_index = 2
                    fig, ax = plt.subplots(nrows=1, ncols=len(levels_ind) - 2)
                else:
                    drop_index = 1
                    fig, ax = plt.subplots(nrows=1, ncols=len(levels_ind) - 1)
            else:
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
            if not x.size or not y.size:
                continue
            if plot_all_data_points:
                plt.scatter(x, y, label=i + 1, alpha=0.2)
                plt.axhline(y=0, color='k')
                plt.axvline(x=0, color='k')
            else:
                slope, intercept, r, p, stderr = stats.linregress(x, y)
                x, y = _data2avg_bins(x, y)
                first_data_point_index = _get_requested_data_index(len(y),
                                                                   FIRT_DATA_POINTS_PERCENTAGE)
                print('num data points: {} for level {}'.format(len(x), i + 1))
                boundary = _get_axis_boundary_weights_ratio(weights_arr)

                sub_plot_title = r"$\bf{" + "Level: " + str(
                    i + 1) + "}$" + "\npvalue={1}\nstderr={2}\nrvalue={3}".format(
                    i + 1, "%.2f" % p, "%.2f" % r, "%.2f" % stderr)

                current_fig = fig.add_subplot(1,
                                              len(levels_ind) - drop_index,
                                              i + 1)

                ax[i].set_title(sub_plot_title, fontsize=10)

                ax[i].axis('off')
                dot_size = 10
                plt.scatter(x[:first_data_point_index],
                            y[:first_data_point_index], marker='s', c='r',
                            s=dot_size)
                plt.scatter(x[first_data_point_index:],
                            y[first_data_point_index:], linewidth=0,
                            marker='s', c='black', s=dot_size)
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


def _alinged_weight_avg(cur_lvls):
    cur_lvls_copy = cur_lvls.copy()
    max_iter = np.ceil(0.2 * len(cur_lvls_copy))
    start_idx = 0
    avg_lst = []
    # continue until reach 4/5 from origin len
    while (len(cur_lvls_copy) > max_iter):
        len_lst = [len(l) for l in cur_lvls_copy]
        lst_idx_to_del = np.argmin(len_lst)
        min_len = np.min(len_lst)

        # avg to this point
        if min_len - start_idx <= 0:
            start_idx = min_len
            del cur_lvls_copy[lst_idx_to_del]
            continue
        mat = np.vstack(
            [l[start_idx:start_idx + (min_len - start_idx)] for l in
             cur_lvls_copy])
        start_idx = min_len

        # concat avg
        avg_lst.extend(np.average(mat, axis=0).tolist())
        del cur_lvls_copy[lst_idx_to_del]
    return avg_lst


def aligned_weight_per_level():
    data_dicts = load_dict_data()
    for mouse in data_dicts.keys():
        print(mouse)

        weights_arr = data_dicts[mouse]['res_dict']['wMode']
        inner_lvl_start_ind = np.cumsum(
            data_dicts[mouse]['psy_dict']['dayLength'])
        lvl_start_ind = np.cumsum(data_dicts[mouse]['psy_dict']['level_ind'])

        all_lvl_auditory = []
        all_lvl_bias = []
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

            all_lvl_auditory.append(cur_lvl_auditory_lst)
            all_lvl_bias.append(cur_lvl_bias_lst)

        for i, w in enumerate([all_lvl_auditory, all_lvl_bias]):
            fig, ax = plt.subplots(nrows=1, ncols=len(all_lvl_auditory))
            fig.text(0.5, 0.04, s='Trial', ha='center', fontsize=10)
            fig.text(0, 0.95, "Mouse: {}".format(mouse), va='center',
                     fontsize=12)
            if i == 0:
                fig.text(0.04, 0.5, 'Auditory Weight', va='center',
                         rotation='vertical', fontsize=10)
                boundary = _get_axis_boundary_aligned_weights(
                    weights_arr[W_AUDITORY_IND], all_lvl_auditory)
            else:
                fig.text(0.04, 0.5, 'Bias Weight', va='center',
                         rotation='vertical', fontsize=10)
                boundary = _get_axis_boundary_aligned_weights(
                    weights_arr[W_BIAS_IND], all_lvl_auditory)
            for j, lvl in enumerate(w):
                # calc avg for cur lvl
                avg_arr = _alinged_weight_avg(lvl)
                sub_plot_title = r"$\bf{" + "Level: " + str(j + 1) + "}$"

                current_fig = fig.add_subplot(1, len(all_lvl_auditory), j + 1)
                plt.axhline(y=0, color='k')
                plt.axvline(x=0, color='k')
                ax[j].set_title(sub_plot_title, fontsize=10)
                ax[j].axis('off')
                for k, cur_w in enumerate(lvl):
                    first_data_point_index = _get_requested_data_index(
                        len(lvl), FIRT_DATA_POINTS_PERCENTAGE)
                    if not i:
                        # bias plot
                        if k <= first_data_point_index:
                            plt.plot(cur_w, c='black', alpha=0.4, linewidth=3)
                        else:
                            plt.plot(cur_w, c='#A9373B', alpha=0.4,
                                     linewidth=3)
                        plt.plot(avg_arr, c='black', linestyle='--')
                    else:
                        # auditory plot
                        if k <= first_data_point_index:
                            plt.plot(cur_w, c='black', alpha=0.4, linewidth=3)
                        else:
                            plt.plot(cur_w, c='#FAA61A', alpha=0.4,
                                     linewidth=3)
                        plt.plot(avg_arr, c='black', linestyle='--')
                    plt.xlim([boundary[0], boundary[1]])
                    plt.ylim([boundary[2], boundary[3]])

            plt.show()


if __name__ == '__main__':
    plot_weights_ratio(plot_all_data_points=False, run_clustering=False)
    aligned_weight_per_level()
