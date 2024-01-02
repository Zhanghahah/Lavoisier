# import required modules
import os
import rdkit
import re 
from rdkit import Chem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import BayesPowerlaw as bp
import dataframe_image as dfi

def plot_fit(fit1,
                gamma_mean,
                data_label=None,
                data_color=None,
                edge_color=None,
                fit_color=None,
                scatter_size=10,
                line_width=1,
                fit=True,
                log=True,
                xmin=None,
                linestyle='-.'):
    """
    Function for plotting the date as a power law distribution on a log log scale
    along with the best fit.

    parameters
    ----------

    gamma_mean: (float)
        Final exponent used to generate the best fit curve. For best results
        use the mean of posterior samples.

    data_label: (str)
        curve label.

    data color: (str)
        color of the data scatter plot.

    edge_color: (str)
        color of the scatter marker edge.

    fit_color: (str)
        color of the best fit curve.
    
    scatter_size: (int or float)
        scatter marker size.

    line_width: (int or float)
        width of the best fit curve.

    fit: (bool)
        Whether to plot the best fit curve or not (default True).

    log: (bool)
        Whether to plot in the log scale or not (default True).

    returns
    -------

    None.

    """
    if fit1.discrete:
        unique, counts = np.unique(fit1.data, return_counts=True)
        frequency = counts / np.sum(counts)
    else:
        yx = plt.hist(fit1.data, bins=1000, normed=True)
        plt.clf()
        counts_pre = (yx[0])
        unique_pre = ((yx[1])[0:-1])
        unique = unique_pre[counts_pre != 0]
        frequency = counts_pre[counts_pre != 0]
        # frequency = counts / np.sum(counts)
    X, Y = fit1.powerlawpdf(gamma_mean, xmin)
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.scatter(unique, frequency, s=scatter_size,
                color=data_color, edgecolor=edge_color,label=data_label)
    if fit:
        #X = X[:10]
        #Y = Y[:10]
        plt.plot(X, Y, color=fit_color, linewidth=line_width, linestyle=linestyle)
    return

def describe_vis(data_df, key):
    key_big = key.capitalize()
    key_data_df = data_df[data_df[key] != ""]
    key_data_df = data_df[data_df[key] != '-1']
    key_data_df = data_df[data_df[key].notna()]
    key_data_df[key] = key_data_df[key].apply(lambda x: x.split("."))
    key_data_df[key] = key_data_df[key].apply(lambda x: list(set(map(str.strip, x))))

    key_all = key_data_df[key].explode()

    # combine into single list
    key_list = list(zip(key_all.value_counts().index,
                            key_all.value_counts(),
                            key_all.value_counts(normalize=True)
                        )
                    )

    # create reagent dataframe for processing
    key_columns = [key, 'count', 'frequency']
    key_df = pd.DataFrame(key_list, columns=key_columns)

    # add column for cumulative sum of frequency
    key_df['cumulative'] = key_df.frequency.cumsum()
    key_df_describe = key_df.describe()
    dfi.export(obj=key_df_describe, filename='tmp_figs/powerlaw_describe/{}_woshi_describe.jpg'.format(key_big), table_conversion = 'matplotlib', dpi=500)
    return key_df

def powerlaw_vis(key_df, key):
    key_big = key.capitalize()
    #perform the fitting
    fit=bp.bayes(key_df['count'])
    #get the posterior of exponent attribute. Since we only have a singular power law, we need only the first (index = 0) row of the 2D array.
    posterior=fit.gamma_posterior[0]
    #mean of the posterior is our estimated exponent
    est_exponent=np.mean(posterior)
    print ("alpha: ", est_exponent)

    fig=plt.figure(dpi=90)
    ax1 = fig.add_subplot(111)
    plot_fit(fit, gamma_mean = est_exponent,fit_color='#536694',scatter_size=50,data_color='w',edge_color='#A9D7EB',line_width=2, data_label='Powerlaw Fitting Line of {}'.format(key_big))
    plt.ylim(1*10e-6)
    plt.legend(loc='upper right')
    plt.xlabel('Frequency of {}'.format(key_big))
    plt.ylabel('PDF of {}'.format(key_big))

    #x, y, 长，宽
    axins = fig.add_axes([0.65, 0.55, 0.2, 0.2]) 
    #key_df_filtered = key_df[key_df['count']>2]
    key_df_filtered = key_df
    key_df_filtered = key_df[key_df['count']>100]
    key_df_filtered = key_df_filtered[key_df_filtered['count']<100000]
    x_index = np.linspace(1, len(key_df_filtered), len(key_df_filtered))
    y_counts = key_df_filtered[key_df_filtered.columns[1]]
    axins.bar(x_index,y_counts,label='Number of '+key_big,color='#C6C1D7', width=2)#绘制柱状图
    #plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.5, 1.02), loc=3, borderaxespad=0)
    # plt.ylim(ymax=100000)
    #plt.xticks(rotation=90,fontsize=8)#调整刻度数值显示角度
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    #plt.axes().get_xaxis().set_visible(False)

    fig.show()
    plt.savefig('tmp_figs/powerlaw_describe/{}_woshi_powerlaw_Alpha_{:.2f}_2.svg'.format(key_big, est_exponent),format='svg')

def bar_vis(key_df, key):
    key_big = key.capitalize()
    x_index = key_df[key_df.columns[0]][:25]
            
    y_counts = key_df[key_df.columns[1]][:25]
    y_freq = key_df[key_df.columns[2]][:25]

    fig=plt.figure(dpi=90, figsize=(12, 6))

    font1 = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 10,
            }
    ax1 = fig.add_subplot(111)
    ax1.bar(x_index,y_counts,label='Number of '+key_big,color='#536694', width=0.75)#绘制柱状图
    plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99,0.99), borderaxespad=0, prop = { "size": 13 })

    plt.xticks(rotation=270,fontsize=10)#调整刻度数值显示角度
    plt.yticks(fontsize=12)
    #plt.ylim(60, 1000)
    plt.yscale('log')
    plt.ylabel('Number of '+key_big, fontsize=12)

    ax2 = ax1.twinx()
    ax2.bar(x_index,y_freq,label='Fraction of '+key_big,color='#A9D7EB',  width=0.75)
    plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99, 0.92), borderaxespad=0 ,prop = { "size": 13 })
    plt.xticks(rotation=270,fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 0.14)

    plt.ylabel('Fraction of '+key_big, fontsize=12)
    plt.savefig('tmp_figs/bar/{}_woshi_bar_25_2.svg'.format(key_big),format='svg')

if __name__ == '__main__':
    path = '/data/zhangyu/yuruijie/data/woshi_1000w_v3.csv'
    data_df = pd.read_csv(path)

    feats=['reagents', 'solvents']
    for feat in feats:
        key_df = describe_vis(data_df, feat)
        powerlaw_vis(key_df, feat)
        bar_vis(key_df, feat)

