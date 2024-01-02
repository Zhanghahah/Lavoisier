import pandas as pd
import json
from rdkit import Chem

def load_data(type_here):
    # data_path = 'data/chem_reaction_taxonomy_{}_instruction_test.json'.format(type_here)
    data_path = 'data/bh_reaction_instruction_test.json'
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
    extracted_data_list = []
    
    reaction_template_list = ["Considering a chemical reaction, SMILES is sequenced-based string used to encode the molecular structure. A chemical reaction includes reactants, conditions and products. Thus,reactants for this reaction are ", 
                              ", SMILES for products of reactions are ",
                              ", then the reaction can be described as ",
                              ",reagent for this reaction is ",
                              ", solvent for this reaction is ",]
    
    for i in range(len(data_list)):
        if data_list[i]['type'] != 'catalyst':
            continue
        # print (data_list[i])

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
        extracted_data_list.append([smiles_list[3], smiles_list[4], smiles_list[5]])

    extracted_data_df = pd.DataFrame(extracted_data_list, columns=['reagents', 'solvents', 'catalysts'])
    extracted_data_df.index = [i for i in range(len(extracted_data_df))]
    print (extracted_data_df)
    return extracted_data_df

def data_to_df_bh(data_list):  
    extracted_data_list = []
    # base, solvent, yield, catalyst, reactant, product
    reaction_template_list = ["base for this reaction is ", 
                              ", solvent for this reaction is ",
                              ", product yield for this reaction is ",]
    extracted_data_list = []
    
    for i in range(len(data_list)):
        if data_list[i]['type'] != 'catalyst':
            continue
        cur_data = data_list[i]
        cur_instruct = cur_data['instruction']
        cur_reaction = cur_data['input']
        cur_reaction_s = cur_reaction.split('>>')
        cur_reactants = cur_reaction_s[0].split('.')
        cur_products = cur_reaction_s[1].split('.')
        cur_catalyst = cur_data['output']

        ptr = len(cur_reaction)
        old_ptr = len(cur_reaction)
        feat_list = []
        for i in range(len(reaction_template_list)):
            old_ptr += len(reaction_template_list[i])
            ptr += len(reaction_template_list[i])
            while (cur_instruct[ptr]!=','):
                ptr += 1
            cur_smile = cur_instruct[old_ptr : ptr]
            feat_list.append(cur_smile)
            old_ptr = ptr
        
        extracted_data_list.append([cur_reactants, cur_products, cur_catalyst, 
                                    feat_list[0], feat_list[1], feat_list[2]])

    extracted_data_df = pd.DataFrame(extracted_data_list, columns=['reactants', 'products', 'catalysts',
                                                                   'base', 'solvents', 'yield'])
    extracted_data_df.index = [i for i in range(len(extracted_data_df))]
    print (extracted_data_df)
    return extracted_data_df

def canonical_smiles(smi):
    """
        Canonicalize a SMILES without atom mapping
        """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(
                canonical_smi_list, key=lambda x: (len(x), x)
            )
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi
    
def load_data_cn():
    path = 'data/cn-processed.csv'
    data = pd.read_csv(path)
    data['base_smiles'] = data['base_smiles'].apply(lambda x: canonical_smiles(x))
    data['ligand_smiles'] = data['ligand_smiles'].apply(lambda x: canonical_smiles(x))
    data['substrate_smiles'] = data['substrate_smiles'].apply(lambda x: canonical_smiles(x))
    data['additive_smiles'] = data['additive_smiles'].apply(lambda x: canonical_smiles(x))
    data['product_smiles'] = data['product_smiles'].apply(lambda x: canonical_smiles(x))

    return data

def add_predictions(pred_path, extracted_data_df, top_num):
    predictions = pd.read_csv(pred_path)

    def pred_to_list(predictions):
        pred_list = []
        pred_values = predictions['catalyst_preds'].values
        ptr = 0
        for i in range(int(len(pred_values)/top_num)):
            pred_list.append([[pred_values[j][1:] for j in range(ptr, ptr+top_num)]])
            ptr += top_num
        pred_df = pd.DataFrame(pred_list, columns=['catalyst_preds'])
        return pred_df

    extracted_data_df['catalyst_preds'] = pred_to_list(predictions)
    return extracted_data_df


if __name__ == '__main__':
    type_here_list = []
    # choice = 'bh'
    choice = 'cn'
    for i in range(1):
        type_here_list.append('top_'+str(i))
    top_num = 3
    for type_here in type_here_list:
        if choice == 'cn':
            extracted_data_df = load_data_cn()
        else:
            data_list = load_data(type_here)
            if choice == 'bh':
                extracted_data_df = data_to_df_bh(data_list)
            else:
                extracted_data_df = data_to_df(data_list)

        # pred_path = 'data/11-07_taxonomy_reaction_{}_CN_model_temp1.5_seq{}_greedy_catalyst_preds.csv'.format(type_here[:3]+type_here[-1], top_num)
        # pred_path = 'data/11-26_BH_model_temp1.5_seq3_greedy_catalyst_preds.csv'
        pred_path = 'data/11-07_dolye_CN_model_generate_num_sequences3_greedy_catalyst_preds.csv'
        add_predictions(pred_path, extracted_data_df, top_num)
        # extracted_data_df.to_csv('data/{}_seq{}_src_gt_preds.csv'.format(type_here, top_num))
        # extracted_data_df.to_csv('data/11-26_BH_seq{}_src_gt_preds.csv'.format(top_num))
        extracted_data_df.to_csv('data/cn_seq{}_src_gt_preds.csv'.format(top_num))

        
