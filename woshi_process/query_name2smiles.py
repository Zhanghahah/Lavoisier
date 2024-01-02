import pandas as pd
import os
from tqdm import tqdm

def process_feat(feat):
    if feat == '-1':
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
    res = []
    name_list = []
    smile_list = []
    all_name_list = name_smiles_dict.keys()
    for feat in feat_list:
        if feat in all_name_list:
            res.append(name_smiles_dict[feat])
            name_list.append(feat)
            smile_list.append(name_smiles_dict[feat])
    return res, name_list, smile_list

def load_data(path):
    woshi_path = os.path.join(path, '1000w.csv')
    data = pd.read_csv(woshi_path)
    data.columns = ["id","main_id","reactions_id","reaction_smiles","reference","HeadingText","PargraphText",
                    "conditions","solvents","reagents","yield","temperature","time","pressure","Ph",         
                    "otherconditions","fulltext_of_reaction","Reference_md5_patent_NO","source"]
    data[['reactants', 'products']] = data['reaction_smiles'].str.split('>>', expand=True)

    data2smile_path = os.path.join(path, 'name_to_smiles.csv')
    name_to_smiles_df = pd.read_csv(data2smile_path)
    name_to_smiles_df.columns = ["id","name","smiles"]
    name_to_smiles_df = name_to_smiles_df[(name_to_smiles_df['name'].notna()) &
                                          (name_to_smiles_df['smiles'].notna())]
    name2smiles_dict = name_to_smiles_df.set_index('name')['smiles'].to_dict()
    smiles2name_dict = name_to_smiles_df.set_index('smiles')['name'].to_dict()
    return data, name2smiles_dict, smiles2name_dict

def cal_smiles_map(woshi_data, name2smiles_dict):
    feat_list = ['reagents', 'solvents']
    for feat in feat_list:
        woshi_data[feat] = woshi_data[feat].fillna('-1')
        feats = list(woshi_data[feat].values)
        all_names_num = 0
        all_smiles_num = 0
        all_name_list = []
        all_smiles_list = []
        before_name_list = []
        match_data_num = 0
        for i in tqdm(range(len(feats))):
            if feats[i] == '-1':
                continue
            feats[i] = process_feat(feats[i])
            before_name_list.extend(feats[i])
            all_names_num += len(feats[i])
            name_num_before = len(feats[i])
            feats[i], name_list, smile_list = smiles_map(feats[i], name2smiles_dict)
            match_data_num += (name_num_before == len(feats[i]))
            # print (name_set, len(all_name_set))
            all_smiles_num += len(feats[i])
            all_name_list.extend(name_list)
            all_smiles_list.extend(smile_list)

        print ('current feat: {}'.format(feat))
        print ('there are {} names in the data'.format(all_names_num))
        print ('there are {} distinct names first.'.format(len(set(before_name_list))))
        print ('after matching...')
        print ('there are {} distinct names matched.'.format(len(set(all_name_list))))
        print ('there are {} distinct smiles matched.'.format(len(set(all_smiles_list))))
        print ('there are {} smiles in the data'.format(all_smiles_num))
        print ('there are {} rows matched successfully'.format(match_data_num))
        print ('there are {} not-nan rows all'.format(len(woshi_data[woshi_data[feat]!='-1'])))

if __name__ == '__main__':
    path = '/data/zhangyu/yuruijie/data'
    woshi_data, name2smiles_dict, smiles2name_dict = load_data(path)
    print ('there are {} distinct names in name_to_smile.csv.'.format(len(list(name2smiles_dict.keys()))))
    print ('there are {} distinct smiles in name_to_smile.csv.'.format(len(list(smiles2name_dict.keys()))))

    cal_smiles_map(woshi_data, name2smiles_dict)
    



