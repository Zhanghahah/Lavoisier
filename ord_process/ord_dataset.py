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
from evaluate import canonical_smiles

class RawDataset:
    """
    base class for data process, reaction selection
    """

    def __init__(self, dataset_file_name,
                 root_path_prefix="/data/zhangyu/yuruijie",
                 raw_data_path=""
                 ):
        self.dataset_file_name = dataset_file_name
        self.root_path_prefix = root_path_prefix
        self.raw_data_path = raw_data_path
        self.source_dir_path = os.path.join(
            root_path_prefix,
            raw_data_path,
            dataset_file_name
        )

    @staticmethod
    def reformer_reaction(input_feature):
        re_feat = ""
        for conpound in input_feature.split(";"):
            amount = float(conpound.split(",")[-1].split(":")[-1].strip().split(" ")[0])
            if amount <= 0:
                continue
            re_feat += conpound
            re_feat += ";"

        return re_feat[:-1]

    @staticmethod
    def parse_json(path):
        with open(path) as r:
            data = r.readlines()
        return data

    @staticmethod
    def parse_json_2(path):
        with open(path) as r:
            try:
                data = r.read()
            except:
                print(path)
                return
        return data

    @staticmethod
    def parse_json_3(path):
        print(path)
        with open(path, 'r') as fcc_file:
            data = json.load(fcc_file)
        return data

    def data_formation(self):
        parse_data_format = open(
            os.path.join(
                self.root_path_prefix,
                self.raw_data_path,
                self.dataset_file_name + "_parse_format.json"
            ),
            'w',
            encoding='utf-8'
        )
        data = self.parse_json(
            os.path.join(
                self.dataset_file_name,
                self.root_path_prefix,
                self.raw_data_path
            )
        )














class OpenReactionDataset(RawDataset):
    def __init__(self, dataset_file_name):
        super().__init__(dataset_file_name)

        self.data_path_prefix = os.path.join(
            self.root_path_prefix,
            self.raw_data_path,
            self.dataset_file_name
        )

    def parse_process_data(self):
        ord_path = os.path.join(
            self.data_path_prefix,
            "all_data.json"
        )
        with open(ord_path, 'r') as r:
            data = r.readlines()
        process_data = json.loads(data[0])
        return process_data

    def parse_raw_data(self):
        ord_raw_path = os.path.join(
            self.data_path_prefix,
            'all_result.jsonl'
        )

        with open(ord_raw_path, 'r') as f:
            json_list = list(f)
            # ord_raw_data = json.load(f)
        ord_raw_data = []
        for json_str in json_list:
            result = json.loads(json_str)
            ord_raw_data.append(result)

        return ord_raw_data

    def clear_map_number(smi):
        """Clear the atom mapping number of a SMILES sequence"""
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return canonical_smiles(Chem.MolToSmiles(mol))

    def parse_auto_mapping_data(self):
        """load auto mapping data """
        ord_ap_path = os.path.join(
            self.data_path_prefix,
            'canonicalized_no_dup_trust_0.7.csv'
        )
        ord_ap_data = pd.read_csv(ord_ap_path)
        return ord_ap_data

    def canonical_smiles(self, smi_list):

        """
        Canonicalize a SMILES without atom mapping
        """
        smi = '.'.join(smi_list)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, False
        else:
            canonical_smi = Chem.MolToSmiles(mol)
            # print('>>', canonical_smi)
            if '.' in canonical_smi:
                canonical_smi_list = canonical_smi.split('.')
                canonical_smi_list = sorted(
                    canonical_smi_list, key=lambda x: (len(x), x)
                )
                canonical_smi = '.'.join(canonical_smi_list)
            return canonical_smi, True

    def compound_process(self, type_name, reaction):
        """
        this function is used to extract compound that meet specfic task we defined.

        """
        compound = reaction[type_name]
        compound.sort()

        return compound

    def compound_reformer(self, type_name, reaction):
        """
        this function is for reform compound
        input: str,
        output: list, whether the compounds is not exist
        """
        compounds = reaction[type_name]
        if compounds != '':
            ans = [compound.split(':')[-1] for compound in compounds.split(';')]
        else:
            return None
        return ans


    def single_canonical_smile(self, smi):
        """
            Canonicalize a SMILES without atom mapping
            """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, False
        else:
            canonical_smi = Chem.MolToSmiles(mol)
            # print('>>', canonical_smi)
            if '.' in canonical_smi:
                canonical_smi_list = canonical_smi.split('.')
                canonical_smi_list = sorted(
                    canonical_smi_list, key=lambda x: (len(x), x)
                )
                canonical_smi = '.'.join(canonical_smi_list)
            return canonical_smi, True

    def data_alignment(self):
        # ord_ap_data = self.parse_auto_mapping_data()
        raw_ord_data = self.parse_raw_data()
        process_ord_data = self.parse_process_data()
        print('load data done')

    def data_make_table(self, raw_data):
        """
        input: raw data jsonl
        output: format to dataframe for further investigation
        """
        columns = ['reactants', 'solvents', 'reagents', 'catalysts', 'pure_rxn', 'rxn_smiles', 'yield']
        return

    def distinct_no(self, df):
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


    def parser_instruct_from_raw(self):
        each_query = []
        raw_ord_data = self.parse_raw_data()
        #165w
        total_length = len(raw_ord_data)
        tmp = dict()

        save_file_path = './ord_data.csv'
        save_file_path_all = './ord_data_all_columns.csv'
        save_file_path_77w = './ord_77w.csv'
        # save_file_path_165w = './ord_165w.csv'
        # save_data = pd.DataFrame(columns=['reactants', 'product', 'catalysts', 'reagents', 'solvents', 'yield'])
        # save_data_all = pd.DataFrame(columns=['reactants', 'products', 'catalysts', 'reagents', 'solvents', 'yield', 'reaction_id', 'reaction_name', 'conditions'])
        save_data_all_list = []
        multi_product_num = 0
        low_confidence = 0

        for idx, reaction in tqdm(enumerate(raw_ord_data)):
            # sample case
            """
            {
                'reactants': 'aryl halide:CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)Br;amine:CC(C)N1CCNCC1', 
                'solvents': '', 
                'catalysts': 'metal and ligand:C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=C(C4=CC=CC=C4C=C3)C5=C(C=CC6=CC=CC=C65)P(C7=CC=CC=C7)C8=CC=CC=C8;metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd]', 
                'reagents': 'Base:C(=O)([O-])[O-].[Cs+].[Cs+]', 
                'products': 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C', 
                'reaction_name': '1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination', 
                'reaction_id': 'ord-56b1f4bfeebc4b8ab990b9804e798aa7', 
                'conditions': "To a solution of ethyl 6-bromo-4-(2,4-difluorophenylamino)-7-ethoxyquinoline-3-carboxylate (400 mg, 0.89 mmol) and 1-(Isopropyl)piperazine (254 µl, 1.77 mmol) in dioxane was added cesium carbonate (722 mg, 2.22 mmol), tris(dibenzylideneacetone)dipalladium(0) (40.6 mg, 0.04 mmol) and rac-2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl (55.2 mg, 0.09 mmol). Reaction vessel in oil bath set to 110 °C. 11am  After 5 hours, MS shows product (major peak 499), and SM (minor peak 453).  o/n, MS shows product peak. Reaction cooled, concentrated onto silica, and purified on ISCO. 40g column, 1:1 EA:Hex, then 100% EA.  289mg yellow solid. NMR (EN00180-62-1) supports product, but some oxidised BINAP impurity (LCMS 655).  ", 
                'yield': '0:65.38999938964844', 
                'reference': 'https://chemrxiv.org/engage/chemrxiv/article-details/60c9e3f37792a23bfdb2d471', 
                'main_product': 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C', 
                'main_product_id': 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C', 
                'main_product_yield': '65.38999938964844', 
                'canonicalized_raw_reactants': 'CC(C)N1CCNCC1.CCOC(=O)c1cnc2cc(OCC)c(Br)cc2c1Nc1ccc(F)cc1F', 
                'canonicalized_raw_products': 'CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F', 
                'amap_confidence': 0.6848201063132463, 
                'rxn_with_amap': '[NH:16]1[CH2:17][CH2:18][N:19]([CH:20]([CH3:21])[CH3:22])[CH2:23][CH2:24]1.Br[c:15]1[c:11]([O:12][CH2:13][CH3:14])[cH:10][c:9]2[n:8][cH:7][c:6]([C:4]([O:3][CH2:2][CH3:1])=[O:5])[c:27]([NH:28][c:29]3[cH:30][cH:31][c:32]([F:33])[cH:34][c:35]3[F:36])[c:26]2[cH:25]1>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][n:8][c:9]2[cH:10][c:11]([O:12][CH2:13][CH3:14])[c:15]([N:16]3[CH2:17][CH2:18][N:19]([CH:20]([CH3:21])[CH3:22])[CH2:23][CH2:24]3)[cH:25][c:26]2[c:27]1[NH:28][c:29]1[cH:30][cH:31][c:32]([F:33])[cH:34][c:35]1[F:36]', 
                'rxn_wo_amap': 'CC(C)N1CCNCC1.CCOC(=O)c1cnc2cc(OCC)c(Br)cc2c1Nc1ccc(F)cc1F>>CCOC(=O)c1cnc2cc(OCC)c(N3CCN(C(C)C)CC3)cc2c1Nc1ccc(F)cc1F'}
            """
            
            rxn_smiles = []
            # no need to filter low-yield reactions

            confidence = reaction['amap_confidence']
            if (confidence < 0.7):
                low_confidence += 1
                continue

            rxn_yield = reaction['main_product_yield']
            reactants = reaction['canonicalized_raw_reactants']
            # if len(reactants.split('.')) > 2:
            #     continue

            reagents = self.compound_reformer('reagents', reaction)
            if reagents is not None:
                reagents = [canonical_smiles(x) for x in reagents]
                reagents.sort()
                reagents = '.'.join(reagents)
                rxn_smiles.append(reagents)
                reagent_prompt = f"reagent for this reaction is {reagents}, "
            else:
                reagent_prompt = ""

            catalysts = self.compound_reformer('catalysts', reaction)
            if catalysts is not None:
                catalysts = [canonical_smiles(x) for x in catalysts]
                catalysts.sort()
                catalysts = '.'.join(catalysts)
                rxn_smiles.append(catalysts)
                catalyst_prompt = f"catalyst for this reaction is {catalysts}, "
            else:
                catalyst_prompt = ""

            solvents = self.compound_reformer('solvents', reaction)
            if solvents is not None:
                solvents = [canonical_smiles(x) for x in solvents]
                solvents.sort()
                solvents = '.'.join(solvents)
                rxn_smiles.append(solvents)
                solvent_prompt = f"solvent for this reaction is {solvents}, "
            else:
                solvent_prompt = ""

            # if not reagents and not catalysts and not solvents:
            #     continue

            # filter the data query that the length of reactions are more than 2.
            products = reaction['main_product']
            if len(products.split('.')) > 1:
                multi_product_num += 1
                continue

            save_data_all_list.append([reactants, products, catalysts,
                                                                  reagents, solvents, rxn_yield, 
                                                                  reaction['reaction_id'],
                                                                  reaction['reaction_name'], 
                                                                  reaction['conditions']])
            # save_data_all = save_data_all._append(pd.DataFrame([[reactants, products, catalysts,
            #                                                       reagents, solvents, rxn_yield, 
            #                                                       reaction['reaction_id'],
            #                                                       reaction['reaction_name'], 
            #                                                       reaction['conditions']]], 
            #                                                       columns=save_data_all.columns))
            # save_data = save_data._append(pd.DataFrame([[reactants, products, catalysts, reagents, solvents, rxn_yield]], columns=save_data.columns))
        # save_data_all.to_csv(save_file_path_all)
        # save_data.to_csv(save_file_path)
        save_data_all = pd.DataFrame(save_data_all_list, columns=['reactants', 'product', 
                                    'catalysts', 'reagents', 'solvents', 'yield', 'reaction_id',
                                    'reaction_name', 'conditions'])
        save_data_all.to_csv(save_file_path_77w)

        print ("low confidence num: ", low_confidence)
        print ("multi product num: ", multi_product_num)
        self.distinct_no(save_data_all)
        # low confidence num:  863724
        # multi product num:  15427 



if __name__ == '__main__':
    ord_datasets = OpenReactionDataset("ord-data")
    # ord_datasets.parse_instruct_template()
    ord_datasets.parser_instruct_from_raw()
    # ord_datasets.data_alignment()