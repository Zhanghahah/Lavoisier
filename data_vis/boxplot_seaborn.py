import json
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
import scipy.stats as stats
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings('ignore')

def load_data(type_here, top_num):
    path = 'data/{}_seq{}_src_gt_preds.csv'.format(type_here, top_num)
    data_df = pd.read_csv(path)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda x: eval(x))
    data_df['reagents'] = data_df['reagents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data_df['solvents'] = data_df['solvents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data_df['catalysts'] = data_df['catalysts'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda list_x: ['.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x for x in list_x])

    t = type_here[-1]

    mean_path = '../top-data/top{}.csv'.format(t)
    mean_data = pd.read_csv(mean_path)
    mean_data['reagents'] = mean_data['reagents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    mean_data['solvents'] = mean_data['solvents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    mean_data['catalysts'] = mean_data['catalysts'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)

    def mean_yield(x):
        yield_list = mean_data[(mean_data['reagents']==x['reagents']) &
                             (mean_data['solvents']==x['solvents']) &
                             (mean_data['catalysts']==x['catalysts'])]['yield'].values
        return np.mean(yield_list)
    
    def yield_list(x):
        yield_list = mean_data[(mean_data['reagents']==x['reagents']) &
                             (mean_data['solvents']==x['solvents']) &
                             (mean_data['catalysts']==x['catalysts'])]['yield'].values
        return yield_list
        
    data_df['mean_yield_gt'] = data_df.apply(lambda x: mean_yield(x), axis=1)
    data_df['yield_gt_list'] = data_df.apply(lambda x: yield_list(x), axis=1)
    data_df['catalyst_pred_True'] = data_df.apply(lambda x: x['catalysts'] in x['catalyst_preds'], axis=1)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda x: sorted(x))
    # print (data_df['catalyst_pred_True'], sum(data_df['catalyst_pred_True']))
    print ('Catalyst matching acc: {}/{}'.format(sum(data_df['catalyst_pred_True']), len(data_df)))

    return data_df

def box_plot(data_df, type_here, top_num, figsize=(18, 10)):
    yield_dict = {}
    yield_matching_dict = {}
    yield_matching_dict_fractile = {}
    key1 = 'reagents'
    key2 = 'solvents'
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    sr_pair_num = 0
    max_yield_sr = {}
    
    for (key_1, key_2), data in data_df.groupby([key1, key2]):
        yield_dict[key_1+"_"+key_2] = []
        yield_matching_dict[key_1+"_"+key_2] = []
        cur_idx = data.index
        sr_pair_num+=1
        max_yield_sr[key_1+"_"+key_2] = 0
        yield_matching_dict_fractile[key_1+"_"+key_2] = []

        for i in range(len(data)):
            # yield_dict[key_1+"_"+key_2].append(data['mean_yield_gt'].values[i])
            yield_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            max_yield_sr[key_1+"_"+key_2] = max(max_yield_sr[key_1+"_"+key_2], max(data['yield_gt_list'][cur_idx[i]]))

            if data['catalyst_pred_True'][cur_idx[i]] == False:
                continue
            for pred_catalyst in data['catalyst_preds'][cur_idx[i]]:
                pred_catalyst_df = data[data['catalysts']==pred_catalyst]
                if (len(pred_catalyst_df) == 0):
                    continue
                yield_matching_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
                # yield_matching_dict[key_1+"_"+key_2].append(pred_catalyst_df['mean_yield_gt'].values[0])
            
            data['yield_gt_list'][cur_idx[i]].sort()
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(yield_matching_dict[key_1+"_"+key_2][int(75/100*len(yield_matching_dict[key_1+"_"+key_2]))])
            yield_matching_dict_fractile[key_1+"_"+key_2].append(data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])

    max_yield_sr = dict(sorted(max_yield_sr.items(), key=lambda d: d[1], reverse=True))
    yield_gt_df_list = []
    matching_yield_df_list = []
    for key in max_yield_sr.keys():
        yield_gt_df_list.append([key, yield_dict[key], max_yield_sr[key]])
        if len(yield_matching_dict_fractile[key]) == 0:
            continue
        # for i in range(len(yield_matching_dict_fractile[key])):
        #     matching_yield_df_list.append([key, yield_matching_dict_fractile[key][i], max_yield_sr[key]])

        matching_yield_df_list.append([key, max(yield_matching_dict_fractile[key]), max_yield_sr[key]])
    yield_gt_df = pd.DataFrame(yield_gt_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    yield_gt_df.index = [i for i in range(len(yield_gt_df))]
    matching_yield_df = pd.DataFrame(matching_yield_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    matching_yield_df.index = [i for i in range(len(matching_yield_df))]

    fig=plt.figure(dpi=100, figsize=(len(matching_yield_df), 15))
    ax = fig.add_subplot(facecolor='white')

    if (type_here == 'top_1'):
        reaction = 'Fc1ccccc1Br.Cn1cnc(C#N)c1>>N#CC1=C(C2=CC=CC=C2F)N(C)C=N1'
        t = 0
    elif (type_here == 'top_3'):
        reaction = 'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1.Ic1ccc2ncccc2c1>>CC(C=C1)=C(C2=CC=C(N=CC=C3)C3=C2)C4=C1N(C5OCCCC5)N=C4'
        t = 3
    color = 'black'

    yield_gt_df = yield_gt_df.explode(['yield_gt_list'])

    # print (yield_gt_df)
    # boxprops = dict(linestyle='-', linewidth=2, color='white', facecolor='red')
    whiskerprops = dict(linestyle='-', linewidth=2, color='black')
    # 中位线
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    # 断点
    capprops = dict(linestyle='-', linewidth=2, color='black')
    sns.boxplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, color='lightgrey', linewidth=3, width=0.5, whis=500,)
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, color='grey', jitter=0.2, s=6)
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=matching_yield_df, color='chocolate', jitter=0, s=15, marker='D')
    
    plt.xticks(rotation=90,fontsize=10)

    plt.show()
    plt.savefig('pictures_sns/ord_boxplot_{}_seq{}.svg'.format(type_here, top_num),format='svg', dpi = 150, bbox_inches='tight')
    return

def linear_plot(data_df):
    # data_df = data_df[data_df['catalyst_pred_True']==True]
    print (data_df.columns)
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x.sort())
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x[int(75/100*len(x))])

    sr_list = []
    yield_gt_fractile_list = []
    yield_pred_true_fractile_list = []
    yield_gt_mean_list = []
    yield_pred_true_mean_list = []
    for (key1, key2, key3), data in data_df.groupby(['from_type', 'reagents', 'solvents']):
        sr_list.append(key1+"_"+key2+"_"+key3)
        tmp_yield_list = []
        tmp_yield_pred_true_value = 0
        tmp_yield_pred_true_value_mean = 0
        tmp_yield_pred_true_list = []
        cur_idx = list(data.index)
        for i in range(len(data)):
            tmp_yield_list.extend(data['yield_gt_list'][cur_idx[i]])
            if (data['catalyst_pred_True'][cur_idx[i]] == False):
                continue
            data['yield_gt_list'][cur_idx[i]].sort()
            tmp_yield_pred_true_value = max(tmp_yield_pred_true_value, data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])
            tmp_yield_pred_true_value_mean = max(tmp_yield_pred_true_value_mean, np.mean(data['yield_gt_list'][cur_idx[i]]))
            tmp_yield_pred_true_list.extend(data['yield_gt_list'][cur_idx[i]])
        if (len(tmp_yield_pred_true_list) == 0):
            continue
        tmp_yield_list.sort()
        tmp_yield_pred_true_list.sort()
        yield_gt_fractile_list.append(tmp_yield_list[int(75/100*len(tmp_yield_list))])
        yield_gt_mean_list.append(np.mean(tmp_yield_list))

        # yield_pred_true_fractile_list.append(tmp_yield_pred_true_list[int(75/100*len(tmp_yield_pred_true_list))])
        yield_pred_true_fractile_list.append(tmp_yield_pred_true_value)
        # yield_pred_true_mean_list.append(np.mean(tmp_yield_pred_true_list))
        yield_pred_true_mean_list.append(tmp_yield_pred_true_value_mean)

    # print (yield_gt_fractile_list)
    # print (yield_pred_true_fractile_list)
    beyond_num = sum([yield_pred_true_fractile_list[i]>=yield_gt_mean_list[i] for i in range(len(yield_gt_fractile_list))])
    print ('There are {} points overall. There are {}/{} successful predictions'.format(len(sr_list), beyond_num, len(sr_list)))
    # m, b, r_value, p_value, std_err = stats.linregress(yield_gt_fractile_list, yield_pred_true_fractile_list)
    m, b, r_value, p_value, std_err = stats.linregress(yield_gt_mean_list, yield_pred_true_mean_list)
    # m, b, r_value, p_value, std_err = stats.linregress(yield_gt_mean_list, yield_pred_true_fractile_list)
    print ('Linear Fitting m: {}, r^2: {}, b: {}'.format(m, r_value*r_value, b))
    line_data_df = pd.DataFrame({"Reagent-Solvent 75% Fractile Yield": yield_gt_fractile_list,"Predicted Reagent-Solvent 75% Fractile Yield": yield_pred_true_fractile_list})
    # line_data_df = pd.DataFrame({"Reagent-Solvent Mean Yield": yield_gt_mean_list,"Predicted Reagent-Solvent Mean Yield": yield_pred_true_mean_list})
    # line_data_df = pd.DataFrame({"Reagent-Solvent Mean Yield": yield_gt_mean_list,"Predicted Reagent-Solvent 75% Fractile Yield": yield_pred_true_fractile_list})

    fig=plt.figure(dpi=100, figsize=(18, 18))
    ax = fig.add_subplot(facecolor='white')
    # sns.regplot(x="Reagent-Solvent 75% Fractile Yield", y="Predicted Reagent-Solvent 75% Fractile Yield", data=line_data_df, color='#A5BA49')
    sns.jointplot(x="Reagent-Solvent 75% Fractile Yield", y="Predicted Reagent-Solvent 75% Fractile Yield", data=line_data_df, color='#A5BA49', kind='reg',
                  xlim=(0, 100), ylim=(0, 100))
    # sns.jointplot(x="Reagent-Solvent Mean Yield", y="Predicted Reagent-Solvent Mean Yield", data=line_data_df, color='#A5BA49', kind='reg',
                #   xlim=(0, 100), ylim=(0, 100))
    # sns.jointplot(x="Reagent-Solvent Mean Yield", y="Predicted Reagent-Solvent 75% Fractile Yield", data=line_data_df, color='#A5BA49', kind='reg',
                #   xlim=(0, 100), ylim=(0, 100))
    # ax = sns.displot(data=line_data_df,x='Reagent-Solvent 75% Fractile Yield',kind="kde")
    # ax = sns.displot(data=line_data_df,x='Predicted Reagent-Solvent 75% Fractile Yield',kind="kde")
    

    plt.savefig('pictures_sns/ord_inset_jointplot_{}_seq{}.svg'.format(type_here, top_num),format='svg', dpi = 150, bbox_inches='tight')
    
    # fig2=plt.figure(dpi=100, figsize=(7,7))
    # ax2 = fig2.add_subplot(facecolor='white')
    # # heat map
    # print (line_data_df)
    # bins = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    # # bins = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    # df_index = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    # print (df_index)
    # heat_data_list = []
    # for i in range(len(df_index)):
    #     cur_line = []
    #     tmp_data_df = line_data_df[(line_data_df['Reagent-Solvent 75% Fractile Yield'] >= bins[i]) & (line_data_df['Reagent-Solvent 75% Fractile Yield'] < bins[i+1])]
    #     for j in range(len(df_index)):
    #         cur_line.append(len(tmp_data_df[(tmp_data_df['Predicted Reagent-Solvent 75% Fractile Yield'] >= bins[len(bins)-2-j]) & (tmp_data_df['Predicted Reagent-Solvent 75% Fractile Yield'] < bins[len(bins)-1-j])]))
    #     heat_data_list.append(cur_line)
    # heat_data_df = pd.DataFrame(heat_data_list, columns=df_index)
    # df_index.reverse()
    # heat_data_df.index = df_index
    # print (heat_data_df)
    # sns.heatmap(heat_data_df, cmap='Blues', annot=True)
    # plt.xlabel('Reagent-Solvent 75% Fractile Yield')
    # plt.ylabel('Predicted Reagent-Solvent 75% Fractile Yield')
    # plt.title('Heat map of Reagent-Solvent 75% Fractile Yield')
    # plt.savefig('pictures_sns/ord_inset_heatmap_{}_seq{}.svg'.format(type_here, top_num),format='svg', dpi = 150, bbox_inches='tight')

def heat_plot_top0(data_df):
    
    # print (data_df)
    # print (data_df.columns)

    data_df['reagent_solvent'] = data_df.apply(lambda x: x['reagents']+"_"+x['solvents'], axis=1)
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x.sort())
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x[int(75/100*len(x))])

    # try from_type 0
    for type_here in data_df['from_type'].unique():
        fig2=plt.figure(dpi=100, figsize=(7,6))
        ax2 = fig2.add_subplot(facecolor='white')
        # type_here = 'top_1'
        print ('Running type {}'.format(type_here))
        top_num = 3
        cur_data_df = data_df[data_df['from_type'] == type_here]
        catalysts_list = []
        print ('There are {} unique catalysts.'.format(len(cur_data_df['catalysts'].unique())))
        print ('There are {} unique reagents-solvent pairs.'.format(len(cur_data_df['reagent_solvent'].unique())))
        heat_data_list = []
        for catalyst in cur_data_df['catalysts'].unique():
            cur_row = []
            for rs in cur_data_df['reagent_solvent'].unique():
                tmp_df = cur_data_df[(cur_data_df['catalysts']==catalyst) & (cur_data_df['reagent_solvent']==rs)]
                if len(tmp_df) == 0:
                    cur_row.append(0)
                    continue
                cur_row.append(tmp_df['yield_gt_fractile'].iloc[0])
            heat_data_list.append(cur_row)
        pred_row = []
        for rs in cur_data_df['reagent_solvent'].unique():
            reagent, solvent = rs.split('_')
            tmp_df = cur_data_df[(cur_data_df['reagents']==reagent) & (cur_data_df['solvents']==solvent) & (cur_data_df['catalyst_pred_True']==True)]
            if (len(tmp_df['yield_gt_fractile']) == 0):
                pred_row.append(0)
                continue
            pred_row.append(max(tmp_df['yield_gt_fractile']))
        heat_data_list.append(pred_row)

        heat_data_df = pd.DataFrame(heat_data_list, columns=cur_data_df['reagent_solvent'].unique())
        heat_index = list(cur_data_df['catalysts'].unique())
        heat_index.append('Prediction')
        heat_data_df.index = heat_index

        heat_data_df['mean_yield'] = heat_data_df.apply(lambda x: np.mean([x[catalyst] for catalyst in cur_data_df['reagent_solvent'].unique()]), axis=1)
        heat_data_df = heat_data_df.sort_values(by='mean_yield', ascending=False)
        heat_data_df = heat_data_df.drop(['mean_yield'], axis=1)

        heat_data_df._append(pd.DataFrame([np.mean(heat_data_df[column].values) for column in heat_data_df.columns]))
        sr_mean_yield = [np.mean(heat_data_df[column].values) for column in heat_data_df.columns]
        sr_mean_yield_sortidx = sorted(range(len(sr_mean_yield)), key=lambda k:sr_mean_yield[k], reverse=True)
        heat_data_column = list(heat_data_df.columns)
        print (sr_mean_yield_sortidx)
        heat_data_df = heat_data_df[[heat_data_column[idx] for idx in sr_mean_yield_sortidx]]

        idxes = list(heat_data_df.index)
        reidxes = [idxes[0]] + [idxes[2], idxes[1]] + idxes[3:]
        heat_data_df = heat_data_df.reindex(reidxes)
        idxes = list(heat_data_df.index)

        columns = list(heat_data_df.columns)
        heat_data_df = heat_data_df.drop(idxes[4], axis=0)
        heat_data_df = heat_data_df[[columns[i] for i in range(6)]+[columns[7], columns[6]]]
        heat_data_df = heat_data_df[:10]
        heat_data_df = heat_data_df.drop(idxes[9], axis=0)
        # add annotation
        # print (heat_data_df)
        labels = np.zeros((len(heat_data_df.index), len(heat_data_df.columns)))
        labels = []
        for i in range(len(heat_data_df.index)):
            labels.append([])
        for i in range(len(heat_data_df.columns)):
            sr = heat_data_df.columns[i]
            sr_yield_list = heat_data_df[sr].values
            matching_index = 0
            for j in range(1, len(sr_yield_list)):
                if sr_yield_list[0] == sr_yield_list[j]:
                    matching_index = j
                    labels[j].append(str(int(sr_yield_list[0])))
                    for next_j in range(j+1, len(sr_yield_list)):
                        labels[next_j].append("")
                    break
                else:
                    labels[j].append("")
        print (labels)
        labels = labels[1:]
        # print (idxes)
   
        heat_data_df = heat_data_df.drop('Prediction', axis=0)
        labels = np.asarray(labels).reshape(len(heat_data_df.index), len(heat_data_df.columns))
        print (labels)
        # reidxes = ['Cl[Pd+].C[C@]12C[C@@]3(C)O[C@](C)(C[C@@](C)(O1)P3c1ccccc1)O2.C=C[CH2-]', 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1.Cl[Pd+].C=C[CH2-]']
        # heat_data_df = heat_data_df.reindex(reidxes+list(heat_data_df.index)[2:])
        # print (len(heat_data_df.index))
        # print (len(heat_data_df.columns))
        # print (labels)
        # print (heat_data_df.index)
        # print (heat_data_df.columns)
        sns.heatmap(heat_data_df, annot=labels, fmt="")
        plt.savefig('pictures_sns/heatmap/ord_inset_heatmap_{}_seq{}.svg'.format(type_here, top_num), dpi = 150, bbox_inches='tight')

def heat_plot_top2(data_df):
    data_df['reagent_solvent'] = data_df.apply(lambda x: x['reagents']+"_"+x['solvents'], axis=1)
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x.sort())
    data_df['yield_gt_fractile'] = data_df['yield_gt_list'].apply(lambda x: x[int(75/100*len(x))])
    top_num = 3

    for type_here in data_df['from_type'].unique():
        fig2=plt.figure(dpi=100, figsize=(7,6))
        ax2 = fig2.add_subplot(facecolor='white')

        print ('Running type {}'.format(type_here))
        top_num = 3
        cur_data_df = data_df[data_df['from_type'] == type_here]
        catalysts_list = []
        print ('There are {} unique catalysts.'.format(len(cur_data_df['catalysts'].unique())))
        print ('There are {} unique reagents-solvent pairs.'.format(len(cur_data_df['reagent_solvent'].unique())))
        heat_data_list = []
        for catalyst in cur_data_df['catalysts'].unique():
            cur_row = []
            for rs in cur_data_df['reagent_solvent'].unique():
                tmp_df = cur_data_df[(cur_data_df['catalysts']==catalyst) & (cur_data_df['reagent_solvent']==rs)]
                if len(tmp_df) == 0:
                    cur_row.append(0)
                    continue
                cur_row.append(tmp_df['yield_gt_fractile'].iloc[0])
            heat_data_list.append(cur_row)
        pred_row = []
        for rs in cur_data_df['reagent_solvent'].unique():
            reagent, solvent = rs.split('_')
            tmp_df = cur_data_df[(cur_data_df['reagents']==reagent) & (cur_data_df['solvents']==solvent) & (cur_data_df['catalyst_pred_True']==True)]
            if (len(tmp_df['yield_gt_fractile']) == 0):
                pred_row.append(0)
                continue
            pred_row.append(max(tmp_df['yield_gt_fractile']))
        heat_data_list.append(pred_row)

        heat_data_df = pd.DataFrame(heat_data_list, columns=cur_data_df['reagent_solvent'].unique())
        heat_index = list(cur_data_df['catalysts'].unique())
        heat_index.append('Prediction')
        heat_data_df.index = heat_index

        heat_data_df['mean_yield'] = heat_data_df.apply(lambda x: np.mean([x[catalyst] for catalyst in cur_data_df['reagent_solvent'].unique()]), axis=1)
        heat_data_df = heat_data_df.sort_values(by='mean_yield', ascending=False)
        heat_data_df = heat_data_df.drop(['mean_yield'], axis=1)

        heat_data_df._append(pd.DataFrame([np.mean(heat_data_df[column].values) for column in heat_data_df.columns]))
        sr_mean_yield = [np.mean(heat_data_df[column].values) for column in heat_data_df.columns]
        sr_mean_yield_sortidx = sorted(range(len(sr_mean_yield)), key=lambda k:sr_mean_yield[k], reverse=True)
        heat_data_column = list(heat_data_df.columns)
        heat_data_df = heat_data_df[[heat_data_column[idx] for idx in sr_mean_yield_sortidx]]

        # 7, 1 exchange, 2,5
        idxes = list(heat_data_df.index)
        reidxes = [idxes[0], idxes[7]] + idxes[2:7] + [idxes[1]] + idxes[8:]
        heat_data_df = heat_data_df.reindex(reidxes)
        idxes = list(heat_data_df.index)
        reidxes = idxes[:2] + [idxes[5]] + idxes[3:5] + [idxes[2]] + idxes[6:]
        heat_data_df = heat_data_df.reindex(reidxes)
        idxes = list(heat_data_df.index)
        print (idxes)

        columns = list(heat_data_df.columns)
        heat_data_df = heat_data_df[[columns[3], columns[4], columns[7], columns[9],
                                     columns[13], columns[0], columns[6], columns[1]]]
        heat_data_df = heat_data_df.reindex(idxes[:3]+[idxes[4]]+idxes[8:])
        idxes = list(heat_data_df.index)

        labels = np.zeros((len(heat_data_df.index), len(heat_data_df.columns)))
        labels = []
        for i in range(len(heat_data_df.index)):
            labels.append([])
        for i in range(len(heat_data_df.columns)):
            sr = heat_data_df.columns[i]
            sr_yield_list = heat_data_df[sr].values
            matching_index = 0
            for j in range(len(sr_yield_list)):
                if (j == 3):
                    continue
                if sr_yield_list[3] == sr_yield_list[j]:
                    matching_index = j
                    labels[j].append(str(int(sr_yield_list[0])))
                    for next_j in range(j+1, len(sr_yield_list)):
                        labels[next_j].append("")
                    break
                else:
                    labels[j].append("")
        print (labels)
        labels = labels[:4] + labels[5:]
        heat_data_df = heat_data_df.drop('Prediction', axis=0)
        labels = np.asarray(labels).reshape(len(heat_data_df.index), len(heat_data_df.columns))

        sns.heatmap(heat_data_df, annot=labels, fmt="")
        plt.savefig('pictures_sns/heatmap/ord_inset_heatmap_{}_seq{}.svg'.format(type_here, top_num), dpi = 150, bbox_inches='tight')

def scatter_heat_plot(data_df):
    print (data_df.columns)
    # print (data_df.head(10))
    data_df = data_df.drop(['catalyst_pred_True'], axis=1)
    
    new_data_df = pd.DataFrame([], columns=data_df.columns)
    other_points_list = []
    pred_points_list = []
    for (key1, key2, key3), data in data_df.groupby(['from_type', 'reagents', 'solvents']):
        cur_other_points_list = []
        data['yield_sum'] = data['yield_gt_list'].apply(lambda x: sum(x))
        data = data[data['yield_sum']!=0.0]
        
        data['catalyst_preds'] = data['catalyst_preds'].apply(lambda x: set(x))
        pred_catalysts_set = set()
        for i in data['catalyst_preds'].index:
            pred_catalysts_set = pred_catalysts_set.union(data['catalyst_preds'][i])
            # cur_other_points_list.extend(data['yield_gt_list'][i])
        pred_catalysts_list = list(pred_catalysts_set)
        max_idx = -1
        max_mean_yield = 0
        max_yield_list = []
        for i in range(len(pred_catalysts_list)):
            cur_c_yield = data[data['catalysts'] == pred_catalysts_list[i]]['yield_gt_list'].values
            if (len(cur_c_yield) == 0):
                continue
            cur_c_yield = list(cur_c_yield[0])

            # find out the 75 fractile 
            cur_c_yield.sort()
            cur_c_yield_fractile = cur_c_yield[int(75/100*len(cur_c_yield))]

            if cur_c_yield_fractile > max_mean_yield:
                max_mean_yield = cur_c_yield_fractile
                max_idx = i
                max_yield_list = cur_c_yield
        pred_catalyst = pred_catalysts_list[max_idx]
        for i in data['catalyst_preds'].index:
            if (data['catalysts'][i] == pred_catalyst):
                continue
            cur_other_points_list.extend(data['yield_gt_list'][i])
        data['catalyst_preds_true'] = data.apply(lambda x: x['catalysts']==pred_catalyst, axis=1)
        

        # print (len(max_yield_list), len(cur_other_points_list))
        for i in range(len(max_yield_list)):
            other_points_list.extend(cur_other_points_list)
            pred_points_list.extend([max_yield_list[i]]*len(cur_other_points_list))
        # print (len(other_points_list), len(pred_points_list))
        
        # pred_catalyst_yield_list = data[data['catalysts'] == pred_catalyst][]
        new_data_df = new_data_df._append([data])
    points_df = pd.DataFrame([], columns=['all', 'pred'])
    points_df['all'] = pd.DataFrame(other_points_list)
    points_df['pred'] = pd.DataFrame(pred_points_list)

    bins = [i for i in range(0, 102)]
    df_index = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    # print (df_index)
    heat_data_list = []
    for i in range(len(df_index)):
        cur_line = []
        tmp_data_df = points_df[(points_df['all'] >= bins[i]) & (points_df['all'] < bins[i+1])]
        for j in range(len(df_index)):
            cur_line.append(len(tmp_data_df[(tmp_data_df['pred'] >= bins[len(bins)-2-j]) & (tmp_data_df['pred'] < bins[len(bins)-1-j])]))
        heat_data_list.append(cur_line)
        
    heat_data_df = pd.DataFrame(heat_data_list, columns=df_index)
    df_index.reverse()
    heat_data_df.index = df_index

    fig=plt.figure(dpi=100, figsize=(18, 18))
    ax = fig.add_subplot(facecolor='white')
    # sns.heatmap(points_df, fmt="")
    # sns.heatmap(heat_data_df, vmax=10, vmin=0, mask=heat_data_df<1)
    sns.jointplot(x="all", y="pred", data=points_df, color='red', kind='hist', 
                  xlim=(0, 100), ylim=(0, 100))
    plt.savefig('pictures_sns/jointheat/ord_joint_heat_top0_seq3.svg',dpi = 150, bbox_inches='tight')
    

    # new_data_df.to_csv('check.csv')
    
def box_plot0(data_df, type_here, top_num, figsize=(18, 10)):
    yield_dict = {}
    yield_matching_dict = {}
    yield_matching_dict_fractile = {}
    key1 = 'reagents'
    key2 = 'solvents'
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    sr_pair_num = 0
    max_yield_sr = {}
    pick_catalyst = {}
    data_df.to_csv('check.csv')

    
    for (key_1, key_2), data in data_df.groupby([key1, key2]):
        yield_dict[key_1+"_"+key_2] = []
        yield_matching_dict[key_1+"_"+key_2] = []
        cur_idx = data.index
        sr_pair_num+=1
        max_yield_sr[key_1+"_"+key_2] = 0
        yield_matching_dict_fractile[key_1+"_"+key_2] = 0

        for i in range(len(data)):
            # yield_dict[key_1+"_"+key_2].append(data['mean_yield_gt'].values[i])
            yield_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            max_yield_sr[key_1+"_"+key_2] = max(max_yield_sr[key_1+"_"+key_2], max(data['yield_gt_list'][cur_idx[i]]))

            if data['catalyst_pred_True'][cur_idx[i]] == False:
                continue
            for pred_catalyst in data['catalyst_preds'][cur_idx[i]]:
                pred_catalyst_df = data[data['catalysts']==pred_catalyst]
                if (len(pred_catalyst_df) == 0):
                    continue
                yield_matching_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
                # yield_matching_dict[key_1+"_"+key_2].append(pred_catalyst_df['mean_yield_gt'].values[0])
            
            data['yield_gt_list'][cur_idx[i]].sort()
            if data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))] > yield_matching_dict_fractile[key_1+"_"+key_2]:
                pick_catalyst[key_1+"_"+key_2] = data['catalysts'][cur_idx[i]]
                yield_matching_dict_fractile[key_1+"_"+key_2] = data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))]
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(yield_matching_dict[key_1+"_"+key_2][int(75/100*len(yield_matching_dict[key_1+"_"+key_2]))])
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])

    max_yield_sr = dict(sorted(yield_matching_dict_fractile.items(), key=lambda d: d[1], reverse=True))
    print (pick_catalyst)
    print (len(pick_catalyst))
    
    yield_gt_df_list = []
    matching_yield_df_list = []
    need_idx = [0, 2, 4, 6, 8, 10, 12, 14]
    i = -1
    for key in max_yield_sr.keys():
        i+=1
        if i in need_idx:
            continue
        # if i not in need_idx:
        #     continue
        yield_gt_df_list.append([key, yield_dict[key], max_yield_sr[key]])
        # if len(yield_matching_dict_fractile[key]) == 0:
        #     continue
        # for i in range(len(yield_matching_dict_fractile[key])):
        #     matching_yield_df_list.append([key, yield_matching_dict_fractile[key][i], max_yield_sr[key]])

        # matching_yield_df_list.append([key, max(yield_matching_dict_fractile[key]), max_yield_sr[key]])
        matching_yield_df_list.append([key, yield_matching_dict_fractile[key], max_yield_sr[key]])
    yield_gt_df = pd.DataFrame(yield_gt_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    yield_gt_df.index = [i for i in range(len(yield_gt_df))]
    matching_yield_df = pd.DataFrame(matching_yield_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    matching_yield_df.index = [i for i in range(len(matching_yield_df))]

    fig=plt.figure(dpi=100, figsize= (6, 3))
    ax = fig.add_subplot(facecolor='white')

    color = 'black'

    yield_gt_df = yield_gt_df.explode(['yield_gt_list'])

    # print (yield_gt_df)
    boxprops = dict(linestyle='-', linewidth=2, color=(86/255, 87/255, 86/255))
    whiskerprops = dict(linestyle='-', linewidth=1.5, color=(86/255, 87/255, 86/255))
    # 中位线
    medianprops = dict(linestyle='-', linewidth=1.5, color=(86/255, 87/255, 86/255))
    # 断点
    capprops = dict(linestyle='-', linewidth=1.5, color=(86/255, 87/255, 86/255))
    # sns.boxplot(x='solvent_reagent', y='yield_gt_list', 
    #             data=yield_gt_df, 
    #             palette='Blues', 
    #             whiskerprops = whiskerprops, medianprops = medianprops, capprops = capprops,
    #             # color='lightgrey',
    #             linewidth=1.5, width=0.7, whis=500,)
    # for box in bp['boxes']:
    #     box.set_color((86/255, 87/255, 86/255))
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, 
                  palette='Blues', linewidth=0.0, edgecolor=(86/255, 87/255, 86/255),
                #   color='grey', 
                  jitter=0.0, s=4)
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=matching_yield_df,
                  palette='Blues', linewidth=0.8, edgecolor=(86/255, 87/255, 86/255),
                #    color='chocolate',
                  jitter=0, s=8, marker='D')
    
    # plt.xticks([])
    plt.xticks(rotation=90)
    plt.ylim(-10, 110)
    plt.xlabel('Reagent-Solvent', fontsize=18)
    plt.ylabel('Yield', fontsize=18)

    plt.show()
    plt.savefig('box_v2/ord_boxplot_{}_seq{}_1_nobox.svg'.format(type_here, top_num),dpi = 150, bbox_inches='tight')
    return

def box_plot6(data_df, type_here, top_num, figsize=(18, 10)):
    yield_dict = {}
    yield_matching_dict = {}
    yield_matching_dict_fractile = {}
    key1 = 'reagents'
    key2 = 'solvents'
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    sr_pair_num = 0
    max_yield_sr = {}
    
    for (key_1, key_2), data in data_df.groupby([key1, key2]):
        yield_dict[key_1+"_"+key_2] = []
        yield_matching_dict[key_1+"_"+key_2] = []
        cur_idx = data.index
        sr_pair_num+=1
        max_yield_sr[key_1+"_"+key_2] = 0
        yield_matching_dict_fractile[key_1+"_"+key_2] = 0

        for i in range(len(data)):
            # yield_dict[key_1+"_"+key_2].append(data['mean_yield_gt'].values[i])
            yield_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            max_yield_sr[key_1+"_"+key_2] = max(max_yield_sr[key_1+"_"+key_2], max(data['yield_gt_list'][cur_idx[i]]))

            if data['catalyst_pred_True'][cur_idx[i]] == False:
                continue
            for pred_catalyst in data['catalyst_preds'][cur_idx[i]]:
                pred_catalyst_df = data[data['catalysts']==pred_catalyst]
                if (len(pred_catalyst_df) == 0):
                    continue
                yield_matching_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
                # yield_matching_dict[key_1+"_"+key_2].append(pred_catalyst_df['mean_yield_gt'].values[0])
            
            data['yield_gt_list'][cur_idx[i]].sort()
            yield_matching_dict_fractile[key_1+"_"+key_2] = max(yield_matching_dict_fractile[key_1+"_"+key_2], 
                                                                data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(yield_matching_dict[key_1+"_"+key_2][int(75/100*len(yield_matching_dict[key_1+"_"+key_2]))])
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])

    max_yield_sr = dict(sorted(yield_matching_dict_fractile.items(), key=lambda d: d[1], reverse=True))
    
    yield_gt_df_list = []
    matching_yield_df_list = []
    need_idx = [ 2, 3, 5, 8, 10, 12, 13, 16, 17, 18]
    i = -1
    for key in max_yield_sr.keys():
        i+=1
        if i not in need_idx:
            continue
        yield_gt_df_list.append([key, yield_dict[key], max_yield_sr[key]])
        # if len(yield_matching_dict_fractile[key]) == 0:
        #     continue
        # for i in range(len(yield_matching_dict_fractile[key])):
        #     matching_yield_df_list.append([key, yield_matching_dict_fractile[key][i], max_yield_sr[key]])

        # matching_yield_df_list.append([key, max(yield_matching_dict_fractile[key]), max_yield_sr[key]])
        matching_yield_df_list.append([key, yield_matching_dict_fractile[key], max_yield_sr[key]])
    yield_gt_df = pd.DataFrame(yield_gt_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    yield_gt_df.index = [i for i in range(len(yield_gt_df))]
    matching_yield_df = pd.DataFrame(matching_yield_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
    matching_yield_df.index = [i for i in range(len(matching_yield_df))]

    fig=plt.figure(dpi=100)
    ax = fig.add_subplot(facecolor='white')

    if (type_here == 'top_1'):
        reaction = 'Fc1ccccc1Br.Cn1cnc(C#N)c1>>N#CC1=C(C2=CC=CC=C2F)N(C)C=N1'
        t = 0
    elif (type_here == 'top_3'):
        reaction = 'Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1.Ic1ccc2ncccc2c1>>CC(C=C1)=C(C2=CC=C(N=CC=C3)C3=C2)C4=C1N(C5OCCCC5)N=C4'
        t = 3
    color = 'black'

    yield_gt_df = yield_gt_df.explode(['yield_gt_list'])

    # print (yield_gt_df)
    # boxprops = dict(linestyle='-', linewidth=2, color='white', facecolor='red')
    whiskerprops = dict(linestyle='-', linewidth=1, color='black')
    # 中位线
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    # 断点
    capprops = dict(linestyle='-', linewidth=1, color='black')
    sns.boxplot(x='solvent_reagent', y='yield_gt_list', 
                data=yield_gt_df, 
                palette='plasma_r',
                # color='lightgrey',
                linewidth=1, width=0.6, whis=500,)
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, 
                  palette='plasma_r', linewidth=0.6, edgecolor=(122/255, 96/255, 95/255),
                #   color='grey', 
                  jitter=0.05, s=5)
    sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=matching_yield_df,
                  palette='plasma_r', linewidth=1, edgecolor=(122/255, 96/255, 95/255),
                #    color='chocolate',
                  jitter=0, s=10, marker='D')
    
    plt.xticks([])
    plt.xlabel('Reagent-Solvent', fontsize=18)
    plt.ylabel('Yield', fontsize=18)

    plt.show()
    plt.savefig('box_v2/ord_boxplot_{}_seq{}.svg'.format(type_here, top_num),dpi = 150, bbox_inches='tight')
    return


def box_plot0_nobox(data_df, type_here, top_num, figsize=(18, 10)):
    yield_dict = {}
    yield_matching_dict = {}
    yield_matching_dict_fractile = {}
    key1 = 'reagents'
    key2 = 'solvents'
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    sr_pair_num = 0
    max_yield_sr = {}
    pick_catalyst = {}
    
    for (key_1, key_2), data in data_df.groupby([key1, key2]):
        yield_dict[key_1+"_"+key_2] = []
        yield_matching_dict[key_1+"_"+key_2] = []
        cur_idx = data.index
        sr_pair_num+=1
        max_yield_sr[key_1+"_"+key_2] = 0
        yield_matching_dict_fractile[key_1+"_"+key_2] = 0

        for i in range(len(data)):
            yield_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            max_yield_sr[key_1+"_"+key_2] = max(max_yield_sr[key_1+"_"+key_2], max(data['yield_gt_list'][cur_idx[i]]))

            if data['catalyst_pred_True'][cur_idx[i]] == False:
                continue
            for pred_catalyst in data['catalyst_preds'][cur_idx[i]]:
                pred_catalyst_df = data[data['catalysts']==pred_catalyst]
                if (len(pred_catalyst_df) == 0):
                    continue
                yield_matching_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            data['yield_gt_list'][cur_idx[i]].sort()
            if data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))] > yield_matching_dict_fractile[key_1+"_"+key_2]:
                pick_catalyst[key_1+"_"+key_2] = data['catalysts'][cur_idx[i]]
                yield_matching_dict_fractile[key_1+"_"+key_2] = data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))]
    max_yield_sr = dict(sorted(yield_matching_dict_fractile.items(), key=lambda d: d[1], reverse=True))
    

    need_idx = [0, 1, 3, 5, 7, 9, 11, 13]
    picture_num = 2
    for pic_n in range(picture_num):
        yield_gt_df_list = []
        matching_yield_df_list = []
        need_idx_curr = need_idx[int(pic_n/2*len(need_idx)): int((pic_n+1)/2*len(need_idx))]
        i = -1
        for key in max_yield_sr.keys():
            i+=1
            if i not in need_idx_curr:
                continue
            yield_gt_df_list.append([key, yield_dict[key], max_yield_sr[key]])
            matching_yield_df_list.append([key, yield_matching_dict_fractile[key], max_yield_sr[key]])
        yield_gt_df = pd.DataFrame(yield_gt_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
        yield_gt_df.index = [i for i in range(len(yield_gt_df))]
        matching_yield_df = pd.DataFrame(matching_yield_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
        matching_yield_df.index = [i for i in range(len(matching_yield_df))]

        fig=plt.figure(dpi=100, figsize= (4, 4))
        ax = fig.add_subplot(facecolor='white')

        yield_gt_df = yield_gt_df.explode(['yield_gt_list'])

        sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, 
                     linewidth=0.0, edgecolor=(86/255, 87/255, 86/255),
                    #  palette='Blues',
                      color=(122/255, 172/255, 210/255), 
                    jitter=0.0, s=6)
        sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=matching_yield_df, 
                    linewidth=0.0, edgecolor=(86/255, 87/255, 86/255),
                    color='orange',
                    #   color='grey', 
                    jitter=0.0, s=6)
        
        # plt.xticks([])
        plt.xticks(rotation=90)
        plt.ylim(-10, 110)
        plt.xlabel('Reagent-Solvent', fontsize=18)
        plt.ylabel('Yield', fontsize=18)

        plt.show()
        plt.savefig('box_v2/ord_boxplot_{}_seq{}_nobox_{}.svg'.format(type_here, top_num, pic_n),dpi = 150, bbox_inches='tight')
    return

def box_plot6_nobox(data_df, type_here, top_num, figsize=(18, 10)):
    yield_dict = {}
    yield_matching_dict = {}
    yield_matching_dict_fractile = {}
    key1 = 'reagents'
    key2 = 'solvents'
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    sr_pair_num = 0
    max_yield_sr = {}
    
    for (key_1, key_2), data in data_df.groupby([key1, key2]):
        yield_dict[key_1+"_"+key_2] = []
        yield_matching_dict[key_1+"_"+key_2] = []
        cur_idx = data.index
        sr_pair_num+=1
        max_yield_sr[key_1+"_"+key_2] = 0
        yield_matching_dict_fractile[key_1+"_"+key_2] = 0

        for i in range(len(data)):
            # yield_dict[key_1+"_"+key_2].append(data['mean_yield_gt'].values[i])
            yield_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
            max_yield_sr[key_1+"_"+key_2] = max(max_yield_sr[key_1+"_"+key_2], max(data['yield_gt_list'][cur_idx[i]]))

            if data['catalyst_pred_True'][cur_idx[i]] == False:
                continue
            for pred_catalyst in data['catalyst_preds'][cur_idx[i]]:
                pred_catalyst_df = data[data['catalysts']==pred_catalyst]
                if (len(pred_catalyst_df) == 0):
                    continue
                yield_matching_dict[key_1+"_"+key_2].extend(data['yield_gt_list'][cur_idx[i]])
                # yield_matching_dict[key_1+"_"+key_2].append(pred_catalyst_df['mean_yield_gt'].values[0])
            
            data['yield_gt_list'][cur_idx[i]].sort()
            yield_matching_dict_fractile[key_1+"_"+key_2] = max(yield_matching_dict_fractile[key_1+"_"+key_2], 
                                                                data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(yield_matching_dict[key_1+"_"+key_2][int(75/100*len(yield_matching_dict[key_1+"_"+key_2]))])
            # yield_matching_dict_fractile[key_1+"_"+key_2].append(data['yield_gt_list'][cur_idx[i]][int(75/100*len(data['yield_gt_list'][cur_idx[i]]))])

    max_yield_sr = dict(sorted(yield_matching_dict_fractile.items(), key=lambda d: d[1], reverse=True))

    need_idx = [ 2, 3, 5, 8, 10, 12, 13, 16, 17]
    picture_num = 2
    for pic_n in range(picture_num):
        yield_gt_df_list = []
        matching_yield_df_list = []
        need_idx_curr = need_idx[int(pic_n/2*len(need_idx)): int((pic_n+1)/2*len(need_idx))]
        i = -1
        for key in max_yield_sr.keys():
            i+=1
            if i not in need_idx_curr:
                continue
            yield_gt_df_list.append([key, yield_dict[key], max_yield_sr[key]])
            matching_yield_df_list.append([key, yield_matching_dict_fractile[key], max_yield_sr[key]])
        yield_gt_df = pd.DataFrame(yield_gt_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
        yield_gt_df.index = [i for i in range(len(yield_gt_df))]
        matching_yield_df = pd.DataFrame(matching_yield_df_list, columns=['solvent_reagent', 'yield_gt_list', 'max_yield_gt'])
        matching_yield_df.index = [i for i in range(len(matching_yield_df))]

        fig=plt.figure(dpi=100, figsize= (4, 4))
        ax = fig.add_subplot(facecolor='white')

        yield_gt_df = yield_gt_df.explode(['yield_gt_list'])

        sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=yield_gt_df, 
                     linewidth=0.0, edgecolor=(86/255, 87/255, 86/255),
                    #  palette='Blues',
                      color=(122/255, 172/255, 210/255), 
                    jitter=0.0, s=6)
        sns.stripplot(x='solvent_reagent', y='yield_gt_list', data=matching_yield_df, 
                    linewidth=0.0, edgecolor=(86/255, 87/255, 86/255),
                    color='orange',
                    #   color='grey', 
                    jitter=0.0, s=6)
        
        # plt.xticks([])
        plt.xticks(rotation=90)
        plt.ylim(-10, 110)
        plt.xlabel('Reagent-Solvent', fontsize=18)
        plt.ylabel('Yield', fontsize=18)

        plt.show()
        plt.savefig('box_v2/ord_boxplot_{}_seq{}_nobox_{}.svg'.format(type_here, top_num, pic_n),dpi = 150, bbox_inches='tight')
    return

if __name__ == '__main__':
    type_here_list = []
    for i in range(10):
        type_here_list.append('top_'+str(i))
    # type_here_list = ['top_0']
    type_here_list = ['top_6']
    top_num_list = [3]
    m_list = []
    r2_list = []
    name_list = []

    linear_flag = 0
    linear_data_df = pd.DataFrame([])
    for top_num in top_num_list: 
        for type_here in type_here_list:
            data_df = load_data(type_here, top_num)
            data_df['from_type'] = type_here
            # data_df.to_csv('check.csv')
            linear_data_df = linear_data_df._append(data_df) if len(linear_data_df)!=0 else data_df
            linear_data_df.index = [i for i in range(len(linear_data_df))]
            print ('Current running: {} data, seq{}, Linear Dataframe has a size of {}'.format(type_here, top_num, len(linear_data_df)))
            # print ('data has columns: ', data_df.columns)
            # m, r_value = box_plot(data_df, type_here, top_num, figsize=figsize_dic[type_here])

            box_plot6_nobox(data_df, type_here, top_num)
            # box_plot6(data_df, type_here, top_num)

            # m_list.append(m)
            # r2_list.append(r_value*r_value)
            name_list.append('{}_seq{}'.format(type_here, top_num))

        # if (linear_flag == 1):
        #     continue
        # else:
        #     linear_flag = 0
            # print (linear_data_df)
            # print (sum(linear_data_df['catalyst_pred_True']==True))
        # scatter_heat_plot(linear_data_df)
            # if (type_here == 'top_0'):
            #     heat_plot_top0(linear_data_df)
            # elif (type_here == 'top_2'):
            #     heat_plot_top2(linear_data_df)


            

    # print ('type: ', name_list)
    # print ('m: ', m_list)
    # print ('r^2: ', r2_list)