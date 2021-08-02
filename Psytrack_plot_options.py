from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import psytrack as psy
from Psytrack_sessions_handler import *
from Psytrack_constants import *


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


def save_fig(fig, mouse_name, title, dpi=700, path=None):
    if path is None:
        path = './data/figures/{0}-{1}'.format(mouse_name, title)
    else:
        path = path + "/{}".format(mouse_name)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_res(psy_dict, res_dict, param_dict, mouse_name, mouse_df,
             error_bar=True):
    days = psy_dict["dayLength"]
    days = np.array(days)
    days = days.reshape(days.size, )

    if error_bar:
        fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'],
                               colors=COLORS, zorder=ZORDER,
                               errorbar=res_dict['hess_info']["W_std"], days=days)
    else:
        fig = psy.plot_weights(res_dict['wMode'], param_dict['weights'], days=days,
                               colors=COLORS, zorder=ZORDER)

    all_levels = np.cumsum(get_day_length_by_levels(mouse_df))
    for level in all_levels:
        plt.axvline(level, c='blue', ls='-', lw=0.8, alpha=0.5, zorder=0)

    plt.legend(loc='upper left', prop={'size': 6})

    # add title
    fig.suptitle("Psytrack analysis for Mouse: {}".format(mouse_name))
    save_fig(fig, mouse_name, 'weights')
    plt.show()

    fig_perf = psy.plot_performance(psy_dict)
    save_fig(fig_perf, mouse_name, 'performance')
    # plt.show()

    fig_bias = psy.plot_bias(psy_dict)
    save_fig(fig_bias, mouse_name, 'bias')  # plt.show()
