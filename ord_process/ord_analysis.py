"""
@Time: 2023/11.08
@Author: el_iauk@sjtu.edu.cn
"""
import json
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

def describe_freq_plot(data, key, key_big):
    key_data_df = data[data[key] != '']
    key_data_df = data[data[key] != '-1']
    #key_data_df[key] = key_data_df[key].apply(lambda x: list(set(map(str.strip, x))))

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
    # print (key_df.describe())
    # print (key_df.head(20))
    # print (data[data['reaction']==key_df['reaction'][2]])
    # data[data['reaction']==key_df['reaction'][2]].to_csv('sample.csv')
    
    key_df_describe = key_df.describe()
    dfi.export(obj=key_df_describe, filename='pictures/{}_describe.png'.format(key_big), table_conversion = 'matplotlib', dpi=500)
    dfi.export(obj=key_df.head(10), filename='pictures/{}_sample.png'.format(key_big), table_conversion = 'matplotlib', dpi=500)

    x_index = key_df[key_df.columns[0]]
    y_counts = key_df[key_df.columns[1]]
    y_freq = key_df[key_df.columns[2]]

    '''
    fig=plt.figure(dpi=90, figsize=(12, 6))
    font1 = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 10,
            }
    ax1 = fig.add_subplot(111)
    ax1.bar(x_index,y_counts,label='Number of '+key_big,color='#296073', width=0.75)#绘制柱状图
    plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99,0.99), borderaxespad=0, prop = { "size": 13 })

    plt.xticks(rotation=90,fontsize=10)#调整刻度数值显示角度
    plt.yticks(fontsize=12)
    #plt.ylim(40)
    #plt.ylim(10e3, 10e4)
    #plt.yscale('log')
    plt.ylabel('Number of '+key_big, fontsize=12)
    plt.xlabel(key_big, fontsize=12)
    plt.xticks([])

    # ax2 = ax1.twinx()
    # ax2.bar(x_index,y_freq,label='Fraction of '+key_big,color='#ADC5CF',  width=0.75)
    # plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99, 0.92), borderaxespad=0 ,prop = { "size": 13 })
    # plt.xticks(rotation=90,fontsize=10)
    # plt.yticks(fontsize=12)
    # plt.ylabel('Fraction of '+key_big, fontsize=12)

    plt.savefig('pictures/{}_bar_plot.png'.format(key_big), dpi = 150, bbox_inches='tight')
    # plt.savefig('pictures/{}_bar_plot.svg'.format(key_big),format='svg', dpi = 150, bbox_inches='tight')
    '''


    '''
    x_index = key_df[key_df.columns[0]][:10000]
    y_counts = key_df[key_df.columns[1]][:10000]
    y_freq = key_df[key_df.columns[2]][:10000]

    fig=plt.figure(dpi=90, figsize=(12, 6))
    font1 = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 10,
            }
    ax1 = fig.add_subplot(111)
    ax1.bar(x_index,y_counts,label='Number of '+key_big,color='#296073', width=0.75)#绘制柱状图
    plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99,0.99), borderaxespad=0, prop = { "size": 13 })

    plt.xticks(rotation=90,fontsize=10)#调整刻度数值显示角度
    plt.xticks([])
    plt.yticks(fontsize=12)
    #plt.ylim(40)
    #plt.ylim(10e3, 10e4)
    #plt.yscale('log')
    plt.ylabel('Number of '+key_big, fontsize=12)
    plt.savefig('pictures/{}_bar_plot_all.png'.format(key_big), dpi = 150, bbox_inches='tight')
    '''

    return key_df

def box_plot(freq_df, key1, key2, figsize, dot_size, font_size):
    fig=plt.figure(dpi=100, figsize=figsize)
    idxes = []
    yield_list = []
    yield_dict = {}
    key1_big = key1.capitalize()
    key2_big = key2.capitalize()
    
    for (reaction_, key_1, key_2), data in most_freq_reaction_df.groupby(['reaction', key1, key2]):
        yield_list.append(data['yield'].values)
        idxes.append((key_1+"_"+key_2))
        yield_dict[key_1+"_"+key_2] = data['yield'].values
    
    plt.boxplot([yield_dict[key] for key in yield_dict.keys()], labels=yield_dict.keys(), showfliers=False, vert=False)
    #plt.xticks(rotation=90,fontsize=font_size)#调整刻度数值显示角度
    plt.yticks(fontsize=font_size)#调整刻度数值显示角度
    plt.title("Yield Box Plot in {}-{} pairs".format(key1_big, key2_big))
    # plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99, 0.92), borderaxespad=0 ,prop = { "size": 13 })
    plt.xlabel('Yield', fontsize=12)
    plt.ylabel(key1_big+"_"+key2_big, fontsize=12)
    plt.show()
    plt.savefig('pictures/ord_{}_{}_box_plot.svg'.format(key1_big, key2_big),format='svg', dpi = 150, bbox_inches='tight')

    fig2=plt.figure(dpi=100, figsize=figsize)
    idxes = []
    yield_list = []
    for key in yield_dict.keys():
        idxes.extend([key]*len(yield_dict[key]))
        yield_list.extend(yield_dict[key])
    plt.scatter(x=yield_list, y=idxes, c='w', edgecolors='g', s=dot_size)
    plt.ylabel('Yield', fontsize=12)
    plt.xlabel(key1_big+"_"+key2_big, fontsize=12)
    #plt.xticks(rotation=90,fontsize=font_size)#调整刻度数值显示角度
    plt.yticks(fontsize=font_size)#调整刻度数值显示角度
    plt.title("Yield Scatter Plot in {}-{} pairs".format(key1_big, key2_big))
    plt.show()
    plt.savefig('pictures/ord_{}_{}_scatter_plot.svg'.format(key1_big, key2_big),format='svg', dpi = 150, bbox_inches='tight')

def box_plot_catalyst(most_freq_reaction_df, key, figsize, fontsize):
    fig=plt.figure(dpi=100, figsize=figsize)
    idxes = []
    yield_list = []
    yield_dict = {}
    key_big = key.capitalize()
    
    for (key1), data in most_freq_reaction_df.groupby([key]):
        yield_dict[key1] = data['yield'].values
        # print (len(data['yield'].values), "most")
    

    plt.boxplot([yield_dict[key] for key in yield_dict.keys()], labels=yield_dict.keys(), showfliers=False, vert=False)
    #plt.xticks(rotation=90,fontsize=font_size)#调整刻度数值显示角度
    plt.yticks(fontsize=fontsize)#调整刻度数值显示角度
    plt.title("Yield Box Plot in {}".format(key_big))
    # plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99, 0.92), borderaxespad=0 ,prop = { "size": 13 })
    plt.xlabel('Yield', fontsize=12)
    plt.ylabel(key_big, fontsize=12)
    plt.show()
    plt.savefig('pictures/ord_{}_box_plot.svg'.format(key_big),format='svg', dpi = 150, bbox_inches='tight')

def box_plot_catalyst_all_reactions(freq_df, most_freq_reaction_df, key, figsize, fontsize):
    idxes = []
    yield_list = []
    yield_dict_all = {}
    yield_dict_one = {}
    key_big = key.capitalize()

    most_freq_keys = []
    for (key1), data in most_freq_reaction_df.groupby([key]):
        yield_dict_one[key1] = data['yield'].values
    
    for (key1), data in freq_df.groupby([key]):
        yield_dict_all[key1] = data['yield'].values
    
    # for key in most_freq_keys:
        # print (len(yield_dict[key]), "all")
    def draw_plot(yield_dict1, yield_dict2, edge_color, fill_color, width, line):
        box_ = {"linestyle":line,"linewidth":1.5}
        bp = ax.boxplot([yield_dict1[key] for key in yield_dict2.keys()], labels=list(yield_dict2.keys()), boxprops=box_, patch_artist=True, showfliers=False, vert=False, widths=width)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        
    fig, ax = plt.subplots()
    draw_plot(yield_dict_all, yield_dict_one,  "#ADC5CF", "white", width=0.6, line='..')
    draw_plot(yield_dict_one, yield_dict_one, "#296073", "white", width=0.3, line='--')

    # fig=plt.figure(dpi=100, figsize=figsize)
    # plt.boxplot([yield_dict_all[key] for key in yield_dict_one.keys()], labels=list(yield_dict_one.keys()), showfliers=False, vert=False)
    # plt.boxplot([yield_dict_one[key] for key in yield_dict_one.keys()], labels=list(yield_dict_one.keys()), showfliers=False, vert=False)
    # #plt.xticks(rotation=90,fontsize=font_size)#调整刻度数值显示角度
    plt.yticks(fontsize=fontsize)#调整刻度数值显示角度
    plt.title("Yield Box Plot in {} in most frequent reaction and all reactions".format(key_big))
    # plt.legend(frameon=False,fontsize='medium',bbox_to_anchor=(0.99, 0.92), borderaxespad=0 ,prop = { "size": 13 })
    plt.xlabel('Yield', fontsize=12)
    plt.ylabel(key_big, fontsize=12)
    plt.show()
    plt.legend()
    plt.savefig('pictures/ord_{}_box_plot_all_reactions.svg'.format(key_big),format='svg', dpi = 150, bbox_inches='tight')

def distinct_no(df):
    print ("the no. of all reactions is: {}".format(len(df)))
    key_list = ['reagents', 'solvents', 'catalysts', 'reactants', 'product']
    for key in key_list:
        df[key[:-1]] = df[key].str.split('.')
        tmp_df = df[key[:-1]].explode()
        # print ("the distinct no. of {} is: ".format(key[:-1]), len(tmp_df.unique()))
        print ("the distinct no. of {} combinations is: {}/{}".format(key, len(tmp_df.unique()), 
                                                                        len(df[key].unique())))
    
    df['reaction'] = df.apply(lambda x: x['reactants']+">>"+x['product'], axis=1)
    print ("the distinct no. of reaction is: {}".format(len(df['reaction'] .unique())))


def filter_most_yield(df):
    repeated_num = 0
    final_data = []
    i = 0
    # print (df)
    # df['check'] = df.apply(lambda x: x['reagents']==x['reagents'] and x['solvents']==x['solvents'] and x['catalysts']==x['catalysts'], axis=1)
    # print (len(df[df['check']==0]))
    df['reagents'] = df['reagents'].apply(lambda x: '-1' if x!=x else x)
    df['catalysts'] = df['catalysts'].apply(lambda x: '-1' if x!=x else x)
    df['solvents'] = df['solvents'].apply(lambda x: '-1' if x!=x else x)
    for (key1, key2, key3, key4, key5, key6), data in df.groupby(['reactants', 'product', 'reaction', 'catalysts', 'reagents', 'solvents']):
        mean_yield = np.mean(data['yield'].values)
        i+=len(data['yield'].values)
        final_data.append([key1, key2, key3, 
                            key4, key5, key6, mean_yield])
    final_data = pd.DataFrame(final_data, columns=['reactants', 'product', 'reaction', 'catalysts', 'reagents', 'solvents', 'yield'])
    
    return final_data

    

if __name__ == '__main__':
    # data_path = '/data/zhangyu/yuruijie/ord-data/ord_mean.csv'
    # data_path = '/data/zhangyu/yuruijie/ord-data/ord_distinct.csv'
    # data_path = '/data/zhangyu/yuruijie/ord-data/ord_165w.csv'
    data_path = '/data/zhangyu/yuruijie/ord-data/ord_43w_all_cols.csv'
    data = pd.read_csv(data_path)
    
    print ('data columns are: ', data.columns)
    print ('the data size is {}, while low yield(<50) data size is {}, and high yield(>=50) data size is {}.'.format(len(data), len(data[data['yield']<50]), len(data[data['yield']>=50])) )

    data['reactants'] = data['reactants'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data['reagents'] = data['reagents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data['solvents'] = data['solvents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data['catalysts'] = data['catalysts'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data['reaction'] = data.apply(lambda x: x['reactants']+'>>'+x['product'], axis=1)
    data = filter_most_yield(data)
    # data.to_csv('ord_mean.csv')
    distinct_no(data)


    # data = filter_most_yield(data)
    print ('the filtered data size is {}, while low yield(<50) data size is {}, and high yield(>=50) data size is {}.'.format(len(data), len(data[data['yield']<50]), len(data[data['yield']>=50])) )
    # data.to_csv('ord_distinct.csv')

    freq_df = describe_freq_plot(data, 'reaction', 'Reactions')
    # freq_df = describe_freq_plot(data, 'reagents', 'Reagents')
    
    most_freq_reaction = freq_df['reaction'][0]
    print ('the most diverse reaction is: ', most_freq_reaction)
    # most_freq_reaction_df = data[data['reaction']==most_freq_reaction]
    # box_plot_catalyst(most_freq_reaction_df, 'catalysts', (5, 12), 6)



    # for i in range(10):
    #     most_freq_reaction = freq_df['reaction'][i]
    #     most_freq_reaction_df = data[data['reaction']==most_freq_reaction]
    #     most_freq_reaction_df.to_csv('top-data/top{}.csv'.format(i))
    


    '''
    
    box_plot_catalyst_all_reactions(data, most_freq_reaction_df, 'catalysts', (5, 12), 6)
    box_plot_catalyst(most_freq_reaction_df, 'catalysts', (5, 12), 6)
    distinct_no(data)
    repeated_num = 0
    for (key1, key2, key3, key4), data_ in data.groupby(['reaction', 'reagents', 'solvents', 'catalysts']):
        # print (len(data['yield'].values), "most")
        repeated_num += len(data_['yield'].values) - 1
        # print (data)
        # break
    
    print ("distinct reaction with distinct conditions no.: ", len(data)-repeated_num)
    print ("repeated reaction with conditions no.: ", repeated_num)
    # box_plot_catalyst_compare(data, most_freq_reaction_df, 'catalysts', (5, 12), 6)
    # distinct_no(most_freq_reaction_df)
    

    box_plot(most_freq_reaction_df, 'reagents', 'solvents', (8, 12), 30, 10)
    box_plot(most_freq_reaction_df, 'solvents', 'catalysts', (5, 12), 10, 6)
    box_plot(most_freq_reaction_df, 'catalysts', 'reagents', (5, 12), 10, 6)

    repeated_num = 0
    for (key1, key2, key3, key4), data_ in data.groupby(['reaction', 'reagents', 'solvents', 'catalysts']):
        # print (len(data['yield'].values), "most")
        repeated_num += len(data_['yield'].values) - 1
        # print (data)
        # break
    
    print ("distinct reaction with distinct conditions no.: ", len(data)-repeated_num)
    print ("repeated reaction with conditions no.: ", repeated_num)
    
        

    
    freq_df = describe_freq_plot(data, 'reaction', 'Reactions')

    most_freq_reaction = freq_df['reaction'][0]
    most_freq_reaction_df = data[data['reaction']==most_freq_reaction]
    box_plot(most_freq_reaction_df, 'reagents', 'solvents', (8, 12), 30, 10)
    box_plot(most_freq_reaction_df, 'solvents', 'catalysts', (5, 12), 10, 6)
    box_plot(most_freq_reaction_df, 'catalysts', 'reagents', (5, 12), 10, 6)
    # scatter_plot(most_freq_reaction_df, 'reagents', 'solvents')
    # scatter_plot(most_freq_reaction_df, 'solvents', 'catalysts')
    # scatter_plot(most_freq_reaction_df, 'catalysts', 'reagents')
    box_plot_catalyst(freq_df, 'catalysts', (5, 12), 6)
    '''



        


