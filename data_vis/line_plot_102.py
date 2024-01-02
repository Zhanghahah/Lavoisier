import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import scipy.stats as stats

def process_data(data):
    trainset_idx = ['5', '10', '30', '50', '60', '70']
    res_data = pd.DataFrame([], 
                        columns=['trainset_idx', 
                                 'rf_wo_eh_data_mean', 'rf_wo_eh_data_max', 'rf_wo_eh_data_min',
                                 'rf_5_eh_data_mean', 'rf_5_eh_data_max', 'rf_5_eh_data_min',
                                 'rf_10_eh_data_mean', 'rf_10_eh_data_max', 'rf_10_eh_data_min',
                                 ])
    res_data['trainset_idx'] = trainset_idx
    match_dict = {0.0: 'rf_wo_eh_data',
                  0.05: 'rf_5_eh_data',
                  0.1: 'rf_10_eh_data'
                  }
    
    for key1, data_eh in data.groupby(['data_enhancement_percentage']):
        rf_mean = []
        rf_max = []
        rf_min = []
        prefix_name = match_dict[key1[0]]
        for key2, data_trn in data_eh.groupby(['train_percentage']):
            data_trn_r2 = data_trn['r_squared_test']
            rf_mean.append(np.mean(data_trn_r2))
            rf_max.append(max(data_trn_r2))
            rf_min.append(min(data_trn_r2))
        res_data[prefix_name+'_mean'] = rf_mean
        res_data[prefix_name+'_max'] = rf_max
        res_data[prefix_name+'_min'] = rf_min

    return res_data


def line_plot(data, flag):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    x = data['trainset_idx']
    names = ['rf_wo_eh_data', 'rf_5_eh_data', 'rf_10_eh_data']
    deep_colors = ['#7262ac', '#2e7ebb', '#2e974e']
    shallow_colors = ['#cfcfe5', '#b7d4ea', '#b8e3d2']
    markers = ['X', '^', '8']
    labels = ['RF without enhancement',
              'RF with 0.05 enhancement',
              'RF with 0.1 enhancement',
              ]
    for i in range(len(names)):
        name = names[i]
        ax.fill_between(x, data[name+'_max'], data[name+'_min'], color=shallow_colors[i], alpha=0.3, zorder=0)

        plt.plot(data["trainset_idx"], data[name+'_mean'], linewidth=2, color=deep_colors[i], label=labels[i])
        plt.scatter(data["trainset_idx"], data[name+'_mean'], marker=markers[i], s=50, color=deep_colors[i])
    plt.xlabel('Training Set Volume(%)')
    plt.ylabel( r'$R^2$')
    plt.legend( fontsize=8, frameon=False)

    plt.savefig('./line_102/line_{}.svg'.format(flag))

if __name__ == '__main__':
    path = './data/11_07_post_ft_buchwald-hartwig_enhancenment_results_full.csv'
    path = './data/11-07_new_rxn_suzuki_enhancenment_results_full.csv'
    flag = path.split('/')[-1][:-30]
    data = pd.read_csv(path)
    data = data[['train_percentage', 'data_enhancement_percentage', 'r_squared_test']]
    data = process_data(data)
    line_plot(data, flag)
