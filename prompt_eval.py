# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# @Time: 2023/05/21
# @Author: cynthiazhang
import argparse
import logging
import torch
import sys
import os
import numpy as np
from tqdm import tqdm
import json

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.utils.data import Subset

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__),
            os.path.pardir, "tmp_logs")
        ))

from utils.data.data_utils import get_raw_dataset
from utils.model.model_utils import create_hf_model

logger = logging.getLogger(__name__)

class ChemEval:
    def __init__(self):
        self.output_path_prefix = "/tmp/data_files/"
        self.dataset_name = "yuyuc/chem-uspto"
        self.index_file_name = "yuyuc_chem_uspto_seed1234_train_2,4,4_1.npy"
        self.retrosysthesis = "Retrosynthesis.txt"

    @staticmethod
    def parse_json(path):
        with open(path) as r:
            data = r.readlines()
        return data

    def load_test_from_index_file(self):
        raw_dataset = get_raw_dataset(self.dataset_name, self.output_path_prefix,
                                      None, 0)
        test_dataset = raw_dataset.get_train_data()
        index_file_name = os.path.join(self.output_path_prefix,
                                       self.index_file_name)
        test_index = np.load(index_file_name, allow_pickle=True).tolist()
        test_dataset = Subset(test_dataset, test_index)

        return test_dataset, raw_dataset


def batch_prompt_eval(args, test_dataset, raw_dataset,
                      tokenizer,
                      model_baseline, model_fintuned,
                      base_device, ft_device
                      ):
    for i, tmp_data in enumerate(test_dataset):
        base_prompts = raw_dataset.get_prompt_and_chosen(tmp_data)
        base_inputs = tokenizer(base_prompts, return_tensors="pt").to(base_device)
        ft_prompts = raw_dataset.get_prompt_and_chosen(tmp_data)
        ft_inputs = tokenizer(ft_prompts, return_tensors="pt").to(ft_device)

        print("==========Baseline: Greedy=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          base_inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune: Greedy=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                ft_inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        print("====================prompt end=============================")
        print()
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args,
                model_baseline, model_fintuned,
                tokenizer,
                base_device, ft_device,
                prompts):
    total_length = len(prompts)

    for i in tqdm(range(0, total_length)):
        prompt = prompts[i]
        # tmp_output = dict()
        base_inputs = tokenizer(prompt, return_tensors="pt").to(base_device)
        ft_inputs = tokenizer(prompt, return_tensors="pt").to(ft_device)
        print("==========Baseline: Greedy=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          base_inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune: Greedy=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                ft_inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        # print(r_finetune_g)
        print_utils(r_finetune_g)
        response = r_finetune_g[0].split("Assistant:")[1].split("<|endoftext|>")[0]
        print(response)
        # tmp_output["Reactants"] = reactants[i]
        # tmp_output["Products"] = products[i]
        # tmp_output["Condition"] = response
        # json_str = json.dumps(tmp_output, ensure_ascii=False) + "\n"
        # json_file.write(json_str)

        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        print("====================prompt end=============================")
        print()
        print()


def main():
    args = parse_args()
    finetuned_device = torch.device("cuda:0")
    baseline_device = torch.device("cuda:1")

    # device = torch.device("cuda:1")
    config = AutoConfig.from_pretrained(args.model_name_or_path_baseline)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline,
                                              fast_tokenizer=True)

    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)

    model_baseline.to(baseline_device)
    model_fintuned.to(finetuned_device)

    # chem_eval = ChemEval()
    # test_dataset, raw_dataset = chem_eval.load_test_from_index_file()
    # batch_prompt_eval(args, test_dataset, raw_dataset,
    #                   tokenizer,
    #                   model_baseline, model_fintuned,
    #                   baseline_device, finetuned_device
    #                   )

    raw_instruct_data = ChemEval.parse_json(
        "/home/zhangyu/simple-chem-benchmarks/scripts/llama_chem_v2.json"
    )
    total_test_index = np.load(
        "/home/zhangyu/simple-chem-benchmarks/scripts/test_index.npy",
        allow_pickle=True).tolist()
    test_prompts = []
    for index in total_test_index:
        test_prompts.append("Human: " + \
                            json.loads(raw_instruct_data[index])["INSTRUCTION"] + \
                            " Assistant:")
        if len(test_prompts) > 10:
            break

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    # input_data = chem_eval.parse_json(chem_eval.retrosysthesis)

    if args.language == "English":
        # prompts = []
        reactants = []
        products = []
        # for line in input_data:
        #     input_dict = json.loads(line)
        #     reactant = input_dict["Reactants"]
        #     product = input_dict["Products"]
        #
        #     inputs = f"Human: Here is a chemical reaction formula: Reactants are: {reactant}," \
        #              f"Products are: {product}, please give me the reaction condition of this chemical formula? Assistant:"
        #     prompts.append(inputs)
        #     reactants.append(reactant)
        #     products.append(product)
        prompts = test_prompts
        # prompts = [
        #     "Human: Here is a chemical reaction formula: reactants are an aryl or vinyl halide (R-X) and a boronic acid or boronate ester (R'-B(OR)2), The products are a substituted biphenyl or a styrene derivative. please give me the reaction condition of this chemical formula? Assistant:",
        #     "Human: Here is a chemical reaction formula: Reactants are amine:CC(C)N1CCNCC1, aryl halide:CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)Br, Products are CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C, please give me the reaction condition of this chemical formula? Assistant:",
        #     "Human: Here is a chemical reaction formula: Reagents are Base:C(=O)([O-])[O-].[Cs+].[Cs+],Solvents are Solvent:C1COCCO1,Catalysts are metal and ligand:CC1(C2=C(C(=CC=C2)P(C3=CC=CC=C3)C4=CC=CC=C4)OC5=C1C=CC=C5P(C6=CC=CC=C6)C7=CC=CC=C7)C;metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd].Products are 0:CC(C1=CC(=CC2=C1OC(=CC2=O)N3CCOCC3)C(=O)N(C)C)NC4=CC(=CC(=C4)Cl)F, please give me the reaction condition of this chemical formula？ Assistant:",
        #     "Human: Here is a chemical reaction formula: Reagents are Base:C(=O)([O-])[O-].[Cs+].[Cs+],Solvents are Solvent:CN(C)C=O,Catalysts are metal and ligand:CC(C)C1=CC(=C(C(=C1)C(C)C)C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C)C(C)C;metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd].Products are 0:CC(=O)NC1=C(C=CC(=C1)NC2=NC3=C(C=NN3C(=C2)NC4COC4)C#N)C5CC5, please give me the reaction condition of this chemical formula? Assistant:",
        #     "Human: Here is a chemical reaction formula: Reagents are Base:C(=O)([O-])[O-].[Cs+].[Cs+],Solvents are Solvent:C1COCCO1,Catalysts are metal and ligand:C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=C(C4=CC=CC=C4C=C3)C5=C(C=CC6=CC=CC=C65)P(C7=CC=CC=C7)C8=CC=CC=C8;metal and ligand:CC(=O)O.CC(=O)O.[Pd].Products are 0:C1CN(CCC1O)C2=CC=CC(=C2)C(F)(F)F, please give me the reaction condition of this chemical formula? Assistant:",
        #     "Human: Here is a chemical reaction formula: Reagents are Base:CC(C)(C)[O-].[K+],Solvents are Solvent:C1CCOC1,Catalysts are metal and ligand:CC(C)(C)P(C(C)(C)C)C(C)(C)C;metal and ligand:CC(C)(C)P(C(C)(C)C)C(C)(C)C.CC(C)(C)P(C(C)(C)C)C(C)(C)C.[Pd].Products are 0:CC(C)[C@@H]1CN(CCN1C(=O)OC(C)(C)C)C2=CC(=C(C=C2)OC)OC, please give me the reaction condition of this chemical formula? Assistant:"
        # ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    prompt_eval(args,
                model_baseline, model_fintuned,
                tokenizer,
                baseline_device, finetuned_device,
                prompts)


if __name__ == "__main__":

    main()
