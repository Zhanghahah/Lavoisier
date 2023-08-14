"""
@Time: 2023/08/04
@Author: cynthiazhang@sjtu.edu.cn

this script is for chemistry data process, All datasets involved in this work here:
    USPTO_TPL: classification
    USPTO_MIT: product
    USPTO_50k: reactants
    C–N Coupling
    USPTO_500_MT: mix

Forward reaction prediction: source: reactants.reagents>> target: product (see data/USPTO_500_MT/Product/)
Retrosynthesis: source: product target: reactants (see  data/USPTO_500_MT/Reactants/)
Reagent prediction: source: reactants>>product target: reagents (see data/USPTO_500_MT/Reagents/)
Classification: source: reactants.reagents>>product target: labels (see data/USPTO_500_MT/Classification/)
Yield prediction: source: reactants.reagents>>product target: Yield (see data/USPTO_500_MT/Yield)
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import json
from rdkit import Chem
from reaction_selection import RawDataset, WoshiDataset


class USPTO500MTDataSet(WoshiDataset):
    def __init__(self, dataset_file_name):
        super().__init__(dataset_file_name)

        self.data_path_prefix = os.path.join(
            self.root_path_prefix,
            self.raw_data_path,
            self.dataset_file_name
        )

    def parse_meta_data(self):
        meta_data = pd.read_csv(
            os.path.join(
                self.data_path_prefix,
                "uspto_test.csv"
            )
        )
        print(f"load data done.")
        return meta_data

    def parse_instruct_template(self):
        total_save_name = open(f"/home/zhangyu/data/woshi/chem_uspto_retro_instruction_test.json", 'w', encoding='utf-8')

        uspto_mt_data = self.parse_meta_data()
        tmp = dict()
        total_length = len(uspto_mt_data)
        for idx in tqdm(range(0, total_length)):

            reactants, products, reagents = uspto_mt_data.iloc[idx]['reactants'], uspto_mt_data.iloc[idx]['products'], uspto_mt_data.iloc[idx]['reagents']
            # reactants, products, reagents = self.canonical_smiles(reactants), self.canonical_smiles(products), \
            #                             self.canonical_smiles(reagents)
            rxn_yield = uspto_mt_data.iloc[idx]['Yield']
            class_id = uspto_mt_data.iloc[idx]['labels']
            canonical_rxn = uspto_mt_data.iloc[idx]['canonical_rxn']
            # import pudb
            # pudb.set_trace()
            reaction_smiles = '>>'.join([reactants, products])
            # if reagents != "" or reagents is not None:
            #     tmp['instruction'] = np.random.choice(self.condition).strip()
            #     tmp['input'] = reaction_smiles
            #     tmp['output'] = reagents
            #     tmp['type'] = 'reagents'
            #     total_save_name.write(
            #         json.dumps(
            #             tmp,
            #             ensure_ascii=False
            #         ) + "\n"
            #     )


            # # forward prediction
            # tmp['instruction'] = np.random.choice(self.forward_predict).strip()
            # tmp['input'] = reactants
            # tmp['output'] = products
            # tmp['type'] = 'products'
            # total_save_name.write(
            #     json.dumps(
            #         tmp,
            #         ensure_ascii=False
            #     ) + "\n"
            # )
            #
            # retrosynthesis
            tmp['instruction'] = np.random.choice(self.retro_synthesis).strip()
            tmp['input'] = products
            tmp['output'] = reactants
            tmp['type'] = 'reactants'
            total_save_name.write(
                json.dumps(
                    tmp,
                    ensure_ascii=False
                ) + "\n"
            )
            #
            # # classification
            # classifcation_prompt = ["What is the reaction type of this chemical reaction?",
            #                         "What is the classifcation id of this reaction?",
            #                         "Can you classify this reaction as a chemical reaction?",
            #                         "What is the main type of reaction"]
            # tmp['instruction'] = np.random.choice(classifcation_prompt).strip()
            # tmp['input'] = canonical_rxn
            # tmp['output'] = str(class_id)
            # tmp['type'] = 'classification'
            # total_save_name.write(
            #     json.dumps(
            #         tmp,
            #         ensure_ascii=False
            #     ) + "\n"
            # )
            # # yield prediction
            # yield_prompt = ["Can you measure the yield of the reaction?",
            #                 "What is the yield of this chemical reaction?",
            #                 "What is the percentage yield of the reaction?",
            #                 "What is the chemical reaction's yield in terms of products produced?"
            #                 "How much of the desired product was produced during the reaction?",
            #                 "Can you quantify the amount of the reaction's yield?",
            #                 "Can you estimate the reaction's yield based on the amount of reactants used?"]
            # tmp['instruction'] = np.random.choice(yield_prompt).strip()
            # tmp['input'] = canonical_rxn
            # tmp['output'] = str(rxn_yield)
            # tmp['type'] = 'yield'
            # total_save_name.write(
            #     json.dumps(
            #         tmp,
            #         ensure_ascii=False
            #     ) + "\n"
            # )







if __name__ == '__main__':
    uspto_datasets = USPTO500MTDataSet("woshi")
    uspto_datasets.parse_instruct_template()