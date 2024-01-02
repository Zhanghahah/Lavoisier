"""
@Time: 2023/11/27
@Author: cynthiazhang@sjtu.edu.cn

This script is designed for visualization of generated results.
This work is for drawing two distribution about predicted and observed yields.

for chem_reaction_taxonomy_top_0_instruction_test.json

"""
from typing import Dict

import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluate import parse_json, parse_csv, canonical_smiles
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats

class EvaluateHTE():
    def __init__(self, file_name):
        self.file_name = file_name

def confband(xd, yd, a, b, conf=0.95, x=None):
    """
Calculates the confidence band of the linear regression model at the desired confidence
level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
the best-fit regression line. This is not the same as saying it will contain 95% of
the data points.
Arguments:
- conf: desired confidence level, by default 0.95 (2 sigma)
- xd,yd: data arrays
- a,b: linear fit parameters as in y=ax+b
- x: (optional) array with x values to calculate the confidence band. If none is provided, will
  by default generate 100 points in the original x-range of the data.

Returns:
Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
corresponding to the [input] x array.
Usage: 
    lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
    calculates the confidence bands for the given input arrays
    pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the confidence band
References:
1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm

    """
    alpha = 1. - conf  # significance
    n = len(xd)  # data sample size
    xd, yd = np.array(xd), np.array(yd)
    if x is None:
        x = np.linspace(xd.min(), xd.max(), 100)

    # Predicted values (best-fit model)
    y = a * x + b

    # Auxiliary definitions
    sd = scatterfit(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean()) ** 2)
    sx = (x - xd.mean()) ** 2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = stats.t.ppf(1. - alpha / 2., n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1. / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x

def scatterfit(x, y, a=None, b=None):
    """
   Compute the mean deviation of the data about the linear model given if A,B
   (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about
   the best-fit line.

   x,y assumed to be Numpy arrays. a,b scalars.
   Returns the float sd with the mean deviation.

   
    """

    if a == None:
        # Performs linear regression
        a, b, r, p, err = stats.linregress(x, y)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = np.size(x)
    sd = 1. / (N - 2.) * np.sum((y - a * x - b) ** 2)
    sd = np.sqrt(sd)

    return sd

def columns_descirbe(df_data):

        # df = raw_data[['base_smiles', 'ligand_smiles', 'substrate_smiles', 'additive_smiles', 'base_name', 'product_smiles', 'ligand_name',
        #          'substrate_id', 'additive_id', 'yield']]
        # df = df[df['yield'] > 50].reset_index()
        df = df_data[['reactants', 'product',
                       'catalysts', 'reagents', 'solvents',
                        'yield', 'conditions', 'reaction']]

        return df


def preprocess(raw_df):
    groupby_cn = raw_df.groupby(['reactants', 'product', 'catalysts',
                                 'reagents', 'solvents'], as_index=False).agg({'yield': 'mean'})
    for col in list(groupby_cn.columns):
        if col != 'yield':
            groupby_cn[col] = groupby_cn[col].apply(lambda x: canonical_smiles(x))
    return groupby_cn


def yield_pred_func(raw_data_path, yield_pred_path):
    raw_df = pd.read_csv(raw_data_path)
    raw_df = columns_descirbe(raw_df)
    groupby_df = preprocess(raw_df)
    yield_preds = parse_csv(yield_pred_path)
    yield_preds_df = pd.DataFrame()
    yield_preds_df['yield_pred'] = pd.DataFrame([eval(pred[0])[0] for pred in yield_preds])
    update_groupby_df = pd.concat([groupby_df, yield_preds_df], axis=1)
    results = update_groupby_df[['yield', 'yield_pred']].values.tolist()
    return results


def draw_yield_pred_plot(full_pred_points, save_idx):
    xs = np.linspace(0, 100, 100)
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    ax.plot(xs, xs, color='black', linewidth=1)
    pred_x, y_label = zip(*full_pred_points)
    r2 = r2_score(y_label, pred_x)
    ax.scatter(y_label, pred_x, alpha=0.3, lw=0.75, color="#03910B", s=50)
    ax.text(5, 10, str(np.round(r2, 2)), color='black', fontsize = 16) 

    ax.set_xlabel("Observed Yield", fontsize=18)
    ax.set_ylabel("Predicted Yield", fontsize=18)
    plt.title("Validation", fontsize=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    fig_save_path = f'/data/zhangyu/yuruijie/ord-data/result_data/pictures_yield_pred/{save_idx}'
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    plt.savefig(os.path.join(fig_save_path, f"yield_pre_res{data_idx}.png"), dpi=150)
    plt.close()



def cat_pred_func(raw_path, catal_pred_path, top_num=3):

    """
    cata_preds cotains three candicated results for each query.

    """

    raw_df = pd.read_csv(raw_path)
    raw_df = columns_descirbe(raw_df)
    groupby_cn = preprocess(raw_df)
    cat_preds = parse_csv(catal_pred_path)
    cat_preds = [pred[0] for pred in cat_preds]
    assert len(cat_preds) // len(groupby_cn) == 3

    rxns = groupby_cn.to_dict('records')
    rxn_len = len(rxns)
    updated_rxns: dict[str, float] = dict()
    pred_rxn_cats = dict()
    candicates_points = []
    for rxn in rxns:
        rxn_key = '.'.join([rxn['reactants'], rxn['solvents'], rxn['reagents'], \
                           rxn['catalysts'], rxn['product']])
        rxn_yield = float(rxn['yield'])
        if rxn_key not in updated_rxns:
            updated_rxns[rxn_key] = rxn_yield
        else:
            raise NotImplementedError(f'Invalid data formatting')

    for i in tqdm(range(0, rxn_len)):
        rxn = rxns[i]
        cur_pred = cat_preds[i:i + top_num]
        max_pred_yield = 0
        for cat in cur_pred:
            cat = canonical_smiles(cat.strip())
            pred_rxn_key = '.'.join([
                rxn['reactants'], rxn['solvents'], rxn['reagents'], \
                cat, rxn['product']]
            )
            if pred_rxn_key in updated_rxns:
                pred_yield = updated_rxns[pred_rxn_key]
                if pred_yield > max_pred_yield:
                    max_pred_yield = pred_yield
            else:
                continue
        candicates_points.append((rxn['yield'], max_pred_yield))
    return candicates_points


def draw_cat_pred_plot(full_pred_points):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    x, y = zip(*full_pred_points)
    plt.scatter(x, y, s=2, c='k', alpha=0.3)

    plt.xlabel("Observed Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Validation", fontsize=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("./cat_pre_res.png", dpi=150)
    plt.close()

def draw_yield_pred_plot_v2(full_pred_points, save_idx):
    xs = np.linspace(0, 100, 100)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(1, 1, 1)

    pred_x = full_pred_points['labels'].values
    y_label = full_pred_points['preds'].values
    r2 = r2_score(y_label, pred_x)
    print (r2)

    ax.plot(xs, xs, color='black', linewidth=1)
    ax.scatter(pred_x, y_label,
                alpha=0.5, color=(23/255, 63/255, 190/255),
                s=40
                )
    text = r'$R^2 = {}$'.format(str(np.round(r2, 2)))
    ax.text(60, 10, text, color='black', fontsize = 16) 
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')


    # pred_x, y_label = zip(*full_pred_points)
    # r2 = r2_score(y_label, pred_x)
    # ax.scatter(y_label, pred_x, alpha=0.3, lw=0.75, color="#03910B", s=50)
    # ax.text(2, 10, str(np.round(r2, 2)), color='w', fontsize  =16, zorder=3)  ##

    ax.set_xlabel("Observed Yield(%)", fontsize=18)
    ax.set_ylabel("Predicted Yield(%)", fontsize=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    fig_save_path = f'/data/zhangyu/yuruijie/ord-data/result_data/pictures_yield_pred/{save_idx}'
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    plt.savefig(os.path.join(fig_save_path, f"yield_pre_res{data_idx}.svg"))
    plt.close()

def draw_yield_pred_plot_conf(full_pred_points, save_idx, cur_fill_c, cur_line_c, cur_c):
    xs = np.linspace(0, 100, 100)
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(1, 1, 1)

    pred_x = full_pred_points['labels'].values
    y_label = full_pred_points['preds'].values

    mask = (pred_x!=0) & (y_label!=0)
    pred_x_log = np.log10(pred_x[mask])
    y_label_log = np.log10(y_label[mask])
    mask1 = ~np.isnan(pred_x_log) & ~np.isnan(y_label_log)

    x = np.linspace(min(np.min(y_label[mask]), np.min(pred_x[mask]))-5, np.max(y_label[mask])+5, num=100)
    x_log = np.linspace(np.min(y_label_log[mask1]), np.max(y_label_log[mask1]), num=100)

    m_log, b_log, r_value_log, p_value_log, std_err_log = stats.linregress(np.array(y_label_log[mask1]),
                                                        np.array(pred_x_log[mask1]))
    print (f'm_log: ', m_log)
    print (f'b_log: ', b_log)
    print (f'r_value_log: ', r_value_log)
    rmse = np.sqrt(mean_squared_error(y_label[mask], pred_x[mask]))

    # confidence interval
    lcb_log, ucb_log, x_log_ = confband(np.array(y_label_log[mask1]),
                                        np.array(pred_x_log[mask1]), 
                                        m_log, b_log, conf=0.99)
    
    # lcb = np.power(10, lcb_log)
    # ucb = np.power(10, ucb_log)
    # x = np.power(10, x_log_)

    y_est = np.power(x, m_log)*np.power(10, b_log)
    y_err = x.std() * np.sqrt(1 / len(x) +
                                (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    y_err = y_err *2
    # y_err = x_log.std() * np.sqrt(1 / len(x_log) +
    #                             (x_log - x_log.mean()) ** 2 / np.sum((x_log - x_log.mean()) ** 2))
    # y_err = y_err *10 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    # ax.fill_between(x, ucb, lcb, color="#96B6A2", alpha=0.5)
    # ax.fill_between(x, y_est -y_err, y_est + y_err, color=(235/255, 194/255, 189/255), alpha=0.5)
    # plt.plot(x, y_est, '-', color=(215/255, 118/255, 96/255), linewidth=2)
    # plt.plot(x, y_est -y_err, '--', color=(215/255, 118/255, 96/255), linewidth=2)
    # plt.plot(x, y_est + y_err, '--', color=(215/255, 118/255, 96/255), linewidth=2)
    # plt.scatter(y_label[mask], pred_x[mask], s=50, alpha=0.5, c=(215/255, 118/255, 96/255))

    ax.fill_between(x, y_est -y_err, y_est + y_err, color=(cur_fill_c[0]/255, cur_fill_c[1]/255, cur_fill_c[2]/255), alpha=0.5)
    plt.plot(x, y_est, '-', color=(cur_line_c[0]/255, cur_line_c[1]/255, cur_line_c[2]/255), linewidth=2)
    plt.plot(x, y_est -y_err, '--', color=(cur_line_c[0]/255, cur_line_c[1]/255, cur_line_c[2]/255), linewidth=2)
    plt.plot(x, y_est + y_err, '--', color=(cur_line_c[0]/255, cur_line_c[1]/255, cur_line_c[2]/255), linewidth=2)
    plt.scatter(y_label[mask], pred_x[mask], s=50, alpha=0.5, c=(cur_line_c[0]/255, cur_line_c[1]/255, cur_line_c[2]/255))

    plt.xscale('log')
    plt.yscale('log')

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.set_xlabel("Observed Yield(%)", fontsize=18)
    ax.set_ylabel("Predicted Yield(%)", fontsize=18)
    # plt.title("Validation", fontsize=18)
    
    lim_min = min(np.min(y_label[mask]), np.min(pred_x[mask])) - 1
    lim_max = max(np.max(y_label[mask]), np.max(pred_x[mask])) + 10
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    r2 = r_value_log**2
    text1 = r'$R^2 = {}$'.format(str(np.round(r2, 2)))
    ax.text(15, lim_min+1.3, text1, color='black', fontsize = 16) 
    text2 = r'$RMSE = {}$'.format(str(np.round(rmse, 2)))
    ax.text(15, lim_min+1, text2, color='black', fontsize = 16) 

    print('r2:{}, rsme:{}'.format(np.round(r2, 2), np.round(rmse, 2)))

    # text = r'$R^2 = {} RMSE = {}$'.format(str(np.round(r2, 2)), str(np.round(rmse, 2)))

    plt.tight_layout()
    fig_save_path = f'/data/zhangyu/yuruijie/ord-data/result_data/pictures_yield_pred/{save_idx}'
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    plt.savefig(os.path.join(fig_save_path, f"yield_pre_res{data_idx}_conf_{cur_c}.svg"))
    plt.close()


if __name__ == '__main__':
    total_data_idx = 10
    total_data_idx = 1
    task_type = 'yield' # yield catalyst
    data_mode = 'combine'  # we have two mode including single, combine
    full_pred_points = []
    input_data_prefix = "/home/zhangyu/data/dolye/"
    input_data_prefix = "/data/zhangyu/yuruijie/ord-data/result_data/data/pred_yield_res_2"

    cur_fill_c_list = [(194, 221, 228), (235, 207, 130), (212, 212, 212), (195, 218, 169), (198, 193, 217), (235, 194, 189)]
    cur_line_c_list = [(130, 165, 193), (213, 131, 62), (97, 95, 96), (76, 154, 85), (130, 122, 178), (215, 118, 96)]
    cur_color_list = ["blue", "yellow", "grey", "green", "purple", "red"]

    for data_idx in range(total_data_idx):
        if task_type == 'catalyst':
            catal_pred_file_name = f'11-07_taxonomy_reaction_top{data_idx}_CN_model_temp1.5_seq3_greedy_catalyst_preds.csv'
            raw_file_name = f'top-data/top{data_idx}.csv'
            raw_path, catal_pred_path = os.path.join(input_data_prefix, raw_file_name), \
                                        os.path.join(input_data_prefix, catal_pred_file_name)
            pred_points = cat_pred_func(raw_path, catal_pred_path, top_num=3)
            full_pred_points.extend(pred_points)
            np.save("../tmp_data/full_cat_pred_points.npy",
                    full_pred_points,
                    allow_pickle=True)
        elif task_type == 'yield':
            all_preds = pd.DataFrame([])
            # tmp_pred = '11-26_post_ft_yield_pred_tmp.csv'
            for data_idx in range(1):
            # for data_idx in range(1, 10):
                # tmp_pred = f'11-26_post_ft_yield_pred_{data_idx}.csv'
                tmp_pred = '/data/zhangyu/yuruijie/ord-data/result_data/data/11-26_post_ft_yield_pred_BH.csv'

                tmp_pred = '/data/zhangyu/yuruijie/ord-data/result_data/data/11-07_post_ft_yield_pred_aryl_scope_0.7.csv'
                tmp_pred = '/data/zhangyu/yuruijie/ord-data/result_data/data/11_07_post_ft_yield_pred_aryl_scope_0.7.csv'
                tmp_pred_points = pd.read_csv(tmp_pred)
                tmp_pred_points_re = tmp_pred_points.values.tolist()
                # draw_yield_pred_plot(tmp_pred_points_re, data_idx)
                draw_yield_pred_plot_v2(tmp_pred_points, 'aryl11_07')
                for i in range(6):
                    cur_fill_c = cur_fill_c_list[i]
                    cur_line_c = cur_line_c_list[i]
                    cur_c = cur_color_list[i]
                    draw_yield_pred_plot_conf(tmp_pred_points, 'aryl11_07', cur_fill_c, cur_line_c, cur_c)

                if data_mode == 'combine':
                    all_preds = all_preds._append(tmp_pred_points) if len(all_preds)>0 else tmp_pred_points
                    # all_preds.extend(tmp_pred_points_re)
            
            # draw_yield_pred_plot_v2(all_preds, 'all')
            draw_yield_pred_plot_conf(all_preds, 'all')


            # yield_pred_file_name = f"11-26_post_ft_yield_pred_chem_reaction_taxonomy_top_{data_idx}_instruction_test.csv"
            # raw_file_name = f"top-data/top{data_idx}.csv"
            # if os.path.exists(
            #         os.path.join(input_data_prefix, yield_pred_file_name)
            # ):
            #     raw_path, yield_pred_path = os.path.join(input_data_prefix, raw_file_name), \
            #                                 os.path.join(input_data_prefix, yield_pred_file_name)
            #     pred_points = yield_pred_func(raw_path, yield_pred_path)
            #     draw_yield_pred_plot(pred_points)






