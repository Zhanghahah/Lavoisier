import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import scipy.stats as stats

def line_plot(data):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    x = data['trainset_idx']
    names = ['lav_wo_eh_data', 'lav_w_eh_data', 'rf_wo_eh_data']
    deep_colors = ['#7262ac', '#2e7ebb', '#2e974e']
    shallow_colors = ['#cfcfe5', '#b7d4ea', '#b8e3d2']
    markers = ['X', '^', '8']
    labels = ['Lavoisier without enhancement',
              'Lavoisier with enhancement(1:2)',
              'RF without enhancement']
    for i in range(len(names)):
        # m, b, r_value, p_value, std_err = stats.linregress(np.array(x),
        #                                                     np.array(data[name]))
        # # confidence interval
        # lcb, ucb, x_ = confband(np.array(x),
        #                         np.array(data[name]), 
        #                         m, b, conf=0.99)
        name = names[i]
        ax.fill_between(x, data[name+'_upper'], data[name+'_lower'], color=shallow_colors[i], alpha=0.5)

        plt.plot(data["trainset_idx"], data[name], linewidth=2, color=deep_colors[i], label=labels[i])
        plt.scatter(data["trainset_idx"], data[name], marker=markers[i], s=200, color=deep_colors[i])
    plt.xlabel('Training Set Volume(%)')
    plt.ylabel( r'$R^2$')
    plt.legend( fontsize=8, frameon=False)


    plt.savefig('./line_1228/line.svg')


if __name__ == '__main__':
    lav_wo_eh_data = [-0.01, 0.28, 0.37, 0.44, 0.609, 0.69]
    lav_w_eh_data = [0.01, 0.405, 0.69, 0.73, 0.78, 0.819]
    rf_wo_eh_data = [-0.02, 0.11, 0.20, 0.16, 0.13, 0.28]
    lav_wo_eh_data_upper = [0.0, 0.05, 0.1, 0.08, 0.03, 0.03]
    lav_wo_eh_data_lower = [0.0, 0.06, 0.15, 0.15, 0.15, 0.1]
    lav_w_eh_data_upper = [0.0, 0.05, 0.06, 0.02, 0.02, 0.04]
    lav_w_eh_data_lower = [0.0, 0.08, 0.123, 0.05, 0.05, 0.05]

    trainset_idx = ['5', '10', '30', '50', '60', '70']
    data = pd.DataFrame([], 
                        columns=['trainset_idx', 
                                 'lav_wo_eh_data', 'lav_wo_eh_data_upper', 'lav_wo_eh_data_lower',
                                 'lav_w_eh_data', 'lav_w_eh_data_upper', 'lav_w_eh_data_lower',
                                 'rf_wo_eh_data', 'rf_wo_eh_data_upper', 'rf_wo_eh_data_lower'
                                 ])
    data['trainset_idx'] = trainset_idx
    data['lav_wo_eh_data'] = lav_wo_eh_data
    data['lav_wo_eh_data_upper'] = [lav_wo_eh_data[i] + lav_wo_eh_data_upper[i] for i in range(len(lav_wo_eh_data))]
    data['lav_wo_eh_data_lower'] = [lav_wo_eh_data[i] - lav_wo_eh_data_lower[i] for i in range(len(lav_wo_eh_data))]
    data['lav_w_eh_data'] = lav_w_eh_data
    data['lav_w_eh_data_upper'] = [lav_w_eh_data[i] + lav_w_eh_data_upper[i] for i in range(len(lav_w_eh_data))]
    data['lav_w_eh_data_lower'] = [lav_w_eh_data[i] - lav_w_eh_data_lower[i] for i in range(len(lav_w_eh_data))]
    data['rf_wo_eh_data'] = rf_wo_eh_data
    data['rf_wo_eh_data_upper'] = rf_wo_eh_data
    data['rf_wo_eh_data_lower'] = rf_wo_eh_data
    line_plot(data)