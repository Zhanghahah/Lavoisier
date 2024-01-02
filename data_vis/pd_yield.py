import pandas as pd
import json
import numpy as np

def load_data(type_here):
    data_path = 'data/chem_reaction_taxonomy_{}_instruction_test.json'.format(type_here)
    with open(data_path, 'r') as f:
        json_list = list(f)
    data_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        data_list.append(result)
        # print (result)
    # top-1 len: 896
    print ('{} data here has a size of: '.format(type_here), len(data_list))
    print ('{} reaction is: '.format(type_here), data_list[0]['input'])
    return data_list


def data_to_df(data_list):  
    extracted_data_df = pd.DataFrame(columns=['reaction', 'catalysts', 'reagents', 'yield'])
    
    reaction_template_list = ["Considering a chemical reaction, SMILES is sequenced-based string used to encode the molecular structure. A chemical reaction includes reactants, conditions and products. Thus,reactants for this reaction are ", 
                              ", SMILES for products of reactions are ",
                              ", then the reaction can be described as ",
                              ", catalyst for this reaction is ",
                              ", reagent for this reaction is ",]
    
    for i in range(len(data_list)):
        if data_list[i]['type'] != 'yield':
            continue

        # reactants, products, reaction, reagent, solvent, catalyst
        smiles_list = []
        ptr = 0
        old_ptr = 0
        cur_data = data_list[i]
        cur_instruct = cur_data['instruction']
        for i in range(len(reaction_template_list)):
            old_ptr += len(reaction_template_list[i])
            ptr += len(reaction_template_list[i])
            while (cur_instruct[ptr]!=','):
                ptr += 1
            cur_smile = cur_instruct[old_ptr : ptr]
            smiles_list.append(cur_smile)
            old_ptr = ptr
        smiles_list.append(cur_data['output'])
        # print (smiles_list[3], smiles_list[4], smiles_list[5])
        extracted_data_df = extracted_data_df._append(pd.DataFrame([[smiles_list[2], smiles_list[3], smiles_list[4], smiles_list[5]]], columns=extracted_data_df.columns))

    extracted_data_df['yield'] = extracted_data_df['yield'].astype(np.float64)
    # print (extracted_data_df['yield'])
    extracted_data_df = extracted_data_df[extracted_data_df['yield']<60.0]
    extracted_data_df.index = [i for i in range(len(extracted_data_df))]
    # print (extracted_data_df)
    return extracted_data_df

def add_yield(extracted_data_df, yield_pred_path):
    yield_pred_df = pd.read_csv(yield_pred_path)
    extracted_data_df['yield_pred1'] = 0.0
    extracted_data_df['yield_pred2'] = 0.0
    extracted_data_df['yield_pred3'] = 0.0
    for i in range(len(extracted_data_df)):
        extracted_data_df['yield_pred1'][i] = yield_pred_df['yield_preds'][3*i]
        extracted_data_df['yield_pred2'][i] = yield_pred_df['yield_preds'][3*i+1]
        extracted_data_df['yield_pred3'][i] = yield_pred_df['yield_preds'][3*i+2]    


if __name__ == '__main__':
    type_here_list = []
    for i in range(1, 2):
        type_here_list.append('top_'+str(i))
    top_num = 3
    for type_here in type_here_list:
        data_list = load_data(type_here)
        extracted_data_df = data_to_df(data_list)

        yield_pred_path = 'data/pred_yield_res/11-16_yield_pred_chem_reaction_taxonomy_{}_instruction_test_temp1.5_seq3_greedy_yield_preds.csv'.format(type_here)
        add_yield(extracted_data_df, yield_pred_path)
        extracted_data_df.to_csv('data/pred_yield_res/{}_seq{}_yield.csv'.format(type_here, top_num))