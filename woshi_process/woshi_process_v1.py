import os
import rdkit
import re
from rdkit import Chem
import pandas as pd
import numpy as np
from canonical_smiles import canonical_smiles
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def load_data(path):
    data_path = os.path.join(path, 'woshi_1000w.csv')
    data = pd.read_csv(data_path)
    data.columns = ["id","main_id","reactions_id","reaction_smiles","reference","HeadingText","PargraphText",
                    "conditions","solvents","reagents","yield","temperature","time","pressure","Ph",         
                    "otherconditions","fulltext_of_reaction","Reference_md5_patent_NO","source"]
    data[['reactants', 'products']] = data['reaction_smiles'].str.split('>>', expand=True)

    data2smile_path = os.path.join(path, 'woshi_name_to_smiles.csv')
    name_to_smiles_df = pd.read_csv(data2smile_path)
    name_to_smiles_df.columns = ["id","name","smiles"]
    name_to_smiles_df = name_to_smiles_df[(name_to_smiles_df['name'].notna()) &
                                          (name_to_smiles_df['smiles'].notna())]
    name2smiles_dict = name_to_smiles_df.set_index('name')['smiles'].to_dict()
    smiles2name_dict = name_to_smiles_df.set_index('smiles')['name'].to_dict()
    
    return data, name2smiles_dict, smiles2name_dict

def process_feat(feat):
    if feat == "":
        return feat
    cur_feat_list = []
    feat_s = feat.split(';')
    for cur_feat in feat_s:
        cur_feat_s = cur_feat.split('||')
        i = 0
        while (i < len(cur_feat_s) and cur_feat_s[i]==""):
            i += 1
        if i >= len(cur_feat_s):
            continue
        cur_feat_list.append(cur_feat_s[i])
    cur_feat_list = list(set(map(str.strip, cur_feat_list)))
    return cur_feat_list

def smiles_map(feat_list, name_smiles_dict):
    if feat_list == "":
        return feat_list
    res = []
    name_set = name_smiles_dict.keys()
    for feat in feat_list:
        if feat in name_set:
            res.append(name_smiles_dict[feat])
        else:
            return ""
    return res

def canonical(feat_list):
    if feat_list == "":
        return feat_list
    feat_list = canonical_smiles(feat_list)
    return feat_list

def eval_(x):
    if x=="":
        return x
    return eval(x)

# process woshi data (version 1)
# process woshi data (version 2) change canonical_smiles func
# process woshi data (version 3) take none as "" instead of '-1'
def process_v1_1():
    path = '/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data/raw'
    print ('load data...')
    woshi_data, name2smiles_dict, smiles2name_dict = load_data(path)

    feat_list = ['reagents', 'solvents']
    for feat in feat_list:
        print (f'process feat: {feat}...')
        woshi_data[feat] = woshi_data[feat].fillna("")
        woshi_data[feat] = woshi_data[feat].apply(lambda x: process_feat(x))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: smiles_map(x, name2smiles_dict))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: canonical(x))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: list(set(map(str.strip, x))))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: '.'.join(x) if x!="" else x)

    woshi_data.to_csv(os.path.join('/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data', 'processed/woshi_1000w_v3.csv'))


def process_v1_2():
    path = '/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data/raw'
    print ('load data...')
    woshi_data, name2smiles_dict, smiles2name_dict = load_data(path)
    
    print (woshi_data.columns)
    feat_list = ['reactants', 'products']
    for feat in feat_list:
        print (f'process feat: {feat}...')
        woshi_data[feat] = woshi_data[feat].apply(lambda x: x.split('.'))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: canonical_smiles(x))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: list(set(map(str.strip, x))))
        woshi_data[feat] = woshi_data[feat].apply(lambda x: '.'.join(x) if x!="" else x)

    woshi_data.to_csv(os.path.join('/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data', 'processed/woshi_1000w_v2.csv'))

# create the dictionary from reagent/solvent to reaction ids involving them
def feat2rxn_dict():
    path = os.path.join('/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data',
                        'processed/woshi_1000w_v2.csv')
    woshi_data = pd.read_csv(path)
    woshi_data['reagents'] = woshi_data['reagents'].apply(lambda x: eval_(x))
    woshi_data['solvents'] = woshi_data['solvents'].apply(lambda x: eval_(x))
    reagents = list(woshi_data['reagents'].values)
    ids = list(woshi_data['id'].values)
    reagent2rxnid_dict = {}
    for i in tqdm(range(len(reagents))):
        if reagents[i] == "":
            continue
        for reagent in reagents[i]:
            if reagent2rxnid_dict.get(reagent) == None:
                reagent2rxnid_dict[reagent] = [ids[i]]
            else:
                reagent2rxnid_dict[reagent].append(ids[i])

    solvents = list(woshi_data['solvents'].values)
    solvent2rxnid_dict = {}
    for i in tqdm(range(len(solvents))):
        if solvents[i] == '-1':
            continue
        for solvent in solvents[i]:
            if solvent2rxnid_dict.get(solvent) == None:
                solvent2rxnid_dict[solvent] = [ids[i]]
            else:
                solvent2rxnid_dict[solvent].append(ids[i])
    
    np.save(os.path.join('/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data',
                        'processed/feat2rxnid.npy')
                        , [reagent2rxnid_dict, solvent2rxnid_dict])

def extract_year(x):
    if x!=x:
        return '-1'
    x = x.split(';')
    for i in range(len(x)-1, -1, -1):
        if '20' in x[i] or '19' in x[i] or '18' in x[i]  :
            return x[i]
    return '-1'

def extract_2023_ref():
    path = os.path.join('/home/bml/storage/mnt/v-7db79275c2374/org/yuruijie/data',
                        'processed/woshi_1000w_v2.csv')
    woshi_data = pd.read_csv(path)
    woshi_data['reference'] = woshi_data['reference'].apply(lambda x:extract_year(x))
    print (woshi_data['reference'])
    print (len(woshi_data[woshi_data['reference']!='-1']))
    


if __name__ == "__main__":
    process_v1_1()

