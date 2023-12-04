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
from sklearn.metrics import r2_score

class EvaluateHTE():
    def __init__(self, file_name):
        self.file_name = file_name


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
    ax.text(2, 10, str(np.round(r2, 2)), color='w', fontsize  =16, zorder=3)  ##

    ax.set_xlabel("Observed Yield", fontsize=18)
    ax.set_ylabel("Predicted Yield", fontsize=18)
    plt.title("Validation", fontsize=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    fig_save_path = f'../{save_idx}'
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

if __name__ == '__main__':
    total_data_idx = 10
    task_type = 'yield' # yield catalyst
    data_mode = 'single'  # we have two mode including single, combine
    full_pred_points = []
    input_data_prefix = "/home/zhangyu/data/dolye/"

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
            all_preds = []
            # tmp_pred = '11-26_post_ft_yield_pred_tmp.csv'
            for data_idx in range(1, total_data_idx):
                tmp_pred = f'11-26_post_ft_yield_pred_{data_idx}.csv'
                tmp_pred_points = pd.read_csv(os.path.join(
                    input_data_prefix, tmp_pred
                ))
                tmp_pred_points_re = tmp_pred_points.values.tolist()
                draw_yield_pred_plot(tmp_pred_points_re, data_idx)
                if data_mode == 'combine':
                    all_preds.extend(tmp_pred_points_re)
            if len(all_preds) > 0:
                draw_yield_pred_plot(all_preds, total_data_idx + 1)


            # yield_pred_file_name = f"11-26_post_ft_yield_pred_chem_reaction_taxonomy_top_{data_idx}_instruction_test.csv"
            # raw_file_name = f"top-data/top{data_idx}.csv"
            # if os.path.exists(
            #         os.path.join(input_data_prefix, yield_pred_file_name)
            # ):
            #     raw_path, yield_pred_path = os.path.join(input_data_prefix, raw_file_name), \
            #                                 os.path.join(input_data_prefix, yield_pred_file_name)
            #     pred_points = yield_pred_func(raw_path, yield_pred_path)
            #     draw_yield_pred_plot(pred_points)







