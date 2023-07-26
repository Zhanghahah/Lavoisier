"""
@Time: 2023/07/20
@Author: cynthiazhang@sjtu.edu.cn

this work is for reaction selection
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
from rdkit import Chem
from rxn4chemistry import RXN4ChemistryWrapper


class RawDataset:
    """
    base class for data process, reaction selection
    """

    def __init__(self, dataset_file_name,
                 root_path_prefix="/home/zhangyu",
                 raw_data_path="data"
                 ):
        self.dataset_file_name = dataset_file_name
        self.root_path_prefix = root_path_prefix
        self.raw_data_path = raw_data_path

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


class WoshiDataset(RawDataset):
    def __init__(self, dataset_file_name):
        super().__init__(dataset_file_name)

        self.data_path_prefix = os.path.join(
                            self.root_path_prefix,
                            self.raw_data_path,
                            self.dataset_file_name
                            )
        self.carbo_list = ['C1CCCCO1', 'C1CCCCN1', 'C1CCCN1','C1CCCO1']
        self.rxn_api_key = "apk-423dbe4da7aa41426f3c89ba30464bdd85f56d27c295fd88e7c9661a6e1f734af4b7d66c615c5b5c85a6aa72f5788f2a4ce1a6e9e90aa873be481c0baa348632088fe645cecad8f563991087e7792d09"
        self.rxn_project_id = '64b8f3f767a117001f2953e3'  # O[C@H]1OCC[C@@H](O)[C@@H]1O

        self.rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=self.rxn_api_key)
        self.rxn4chemistry_wrapper.set_project(self.rxn_project_id)

        self.forward_predict = open(
            os.path.join(
                self.data_path_prefix,
                "forward_reaction_prediction_prompt_template.txt"
            ), "r").readlines()

        self.condition = open(
            os.path.join(
                self.data_path_prefix,
                "reagent_prediction_prompt_template.txt"
            ), "r").readlines()

        self.solvents = open(
            os.path.join(
                self.data_path_prefix,
                "solvent_prediction_prompt_template.txt"  # noqa
            ), "r").readlines()

        self.retro_synthesis = open(
            os.path.join(
                self.data_path_prefix,
                "retrosynthesis_prompt_template.txt"
            ), "r").readlines()

        self.reaction_description = open(  # noqa
            os.path.join(
                self.data_path_prefix,
                "reaction_description_prompt_template.txt"
            ), "r").readlines()

    def parse_meta_data(self):
        meta_data = pd.read_csv(
            os.path.join(
                self.data_path_prefix,
                "1000w.csv"
            ), header=None
        )
        print(f"load data done.")
        return meta_data

    def canonical_smiles(self, smi):

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

    def reaction_selection(self, reactants):

        """
        this function is for selection reaction based on specific reactants smiles.(pattern matching)
        selection and save all reactions, all sft templates
        """
        has_subs_match = False
        reactants_list = reactants.split(".")
        for reactant in reactants_list:
            target = Chem.MolFromSmiles(reactant)
            if target is None:
                continue
            for base_carbo in self.carbo_list:
                base = Chem.MolFromSmiles(base_carbo)
                has_subs_match = target.HasSubstructMatch(base)
                if has_subs_match:
                    return True
        return False

    def paragraph_to_actions(self, full_condition):
        """
        this function is for parse description of reaction to actions

        """
        results = self.rxn4chemistry_wrapper.paragraph_to_actions(
            'To a stirred solution of '
            '7-(difluoromethylsulfonyl)-4-fluoro-indan-1-one (110 mg, '
            '0.42 mmol) in methanol (4 mL) was added sodium borohydride '
            '(24 mg, 0.62 mmol). The reaction mixture was stirred at '
            'ambient temperature for 1 hour.'
        )

        return results['actions']
        # print(results['actions'])


def data_proc(data, start_index, end_index, dataset_class):
    results = []
    p_to_actions = []
    for idx in tqdm(range(start_index, end_index)):
        reaction_smiles = data.iloc[idx][3]
        try:
            reactants, products = reaction_smiles.split(">>")
            # solvents, reagents = data.iloc[idx][8], data.iloc[idx][9]
            # reaction_descript = data.iloc[idx][16].strip()
        except:
            continue

        has_match = dataset_class.reaction_selection(reactants)
        if has_match:
            results.append(data.iloc[idx])
            # describe_to_actions = dataset_class.paragraph_to_actions(reaction_descript)
            # p_to_actions.append(describe_to_actions)
    return results


def parse_multi_proc(meta_data, woshi_class):
    meta_data = meta_data.where(pd.notnull(meta_data), None)
    total_length = len(meta_data)
    cpus = 30
    pool = Pool(cpus)
    per_pool_num = total_length // cpus
    for i in range(cpus - 1):
        locals()["process_%s" % i] = pool.apply_async(data_proc, args=(meta_data, i * per_pool_num,
                                                                       i * per_pool_num + per_pool_num, woshi_class))
    locals()["process_%s" % (i + 1)] = pool.apply_async(data_proc,
                                                        args=(meta_data, (i + 1) * per_pool_num, total_length, woshi_class))

    pool.close()
    pool.join()

    total_file_name = open(f"/home/zhangyu/data/woshi/chem_instruct.json", 'w', encoding='utf-8')

    total_reactions = []
    total_actions = []
    for i in range(cpus):
        print(i)
        reactions = locals()["process_{}".format(str(i))].get()
        total_reactions.extend(reactions)
        # total_actions.append(p_to_actions)
    match_nums = len(total_reactions)
    tmp = dict()
    for i in tqdm(range(0, match_nums)):
        reaction_data = total_reactions[i]
        reaction_smiles = reaction_data[3]
        reactants, products = reaction_smiles.split(">>")
        solvents, reagents = reaction_data[8], reaction_data[9]
        # p_to_action = total_actions[i]

        if reagents is not None:
            tmp['instruction'] = np.random.choice(woshi_class.condition).strip()
            tmp['input'] = reaction_smiles
            tmp['output'] = reagents
            total_file_name.write(
                json.dumps(
                    tmp,
                    ensure_ascii=False
                ) + "\n"
            )
        if solvents is not None:
            tmp['instruction'] = np.random.choice(woshi_class.condition).strip()
            tmp['input'] = reaction_smiles
            tmp['output'] = solvents
            total_file_name.write(
                json.dumps(
                    tmp,
                    ensure_ascii=False
                ) + "\n"
            )
        tmp["instruction"] = np.random.choice(woshi_class.retro_synthesis).strip()
        tmp['input'] = products
        tmp['output'] = reactants
        total_file_name.write(
            json.dumps(
                tmp,
                ensure_ascii=False
            ) + "\n"
        )
        tmp["instruction"] = np.random.choice(woshi_class.forward_predict).strip()
        tmp['input'] = reactants
        tmp['output'] = products
        total_file_name.write(
            json.dumps(
                tmp,
                ensure_ascii=False
            ) + "\n"
        )


if __name__ == "__main__":
    ws_datasets = WoshiDataset("woshi")
    woshi_meta_data = ws_datasets.parse_meta_data()
    parse_multi_proc(woshi_meta_data, ws_datasets)






