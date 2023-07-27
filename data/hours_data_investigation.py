"""
@Time: 2023/07/26
@Author: cynthiazhang

this work is for data investigation of hours.

# """
import os
from tqdm import tqdm
import json

from scripts.reaction_selection import RawDataset


def generation_template(input_data, cls):
    parse_ord_qa_to_instruct = open(
            os.path.join(
                cls.root_path_prefix,
                cls.raw_data_path,
                cls.dataset_file_name + "_instruct.json"
            ),
            'w',
            encoding='utf-8'
        )

    total_length = len(input_data)
    for i in tqdm(range(0, total_length)):
        """
        input sample:
        {'reactants': 'amine:CC(C)N1CCNCC1, amount is: 0.001769999973475933 MOLE;aryl halide:CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)Br, amount is: 0.0008859999943524599 MOLE',
         'solvents': '',
         'catalysts': 'metal and ligand:C1=CC=C(C=C1)P(C2=CC=CC=C2)C3=C(C4=CC=CC=C4C=C3)C5=C(C=CC6=CC=CC=C65)P(C7=CC=CC=C7)C8=CC=CC=C8, amount is: 8.859999798005447e-05 MOLE;metal and ligand:C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2.[Pd].[Pd], amount is: 4.4299998990027234e-05 MOLE',
         'reagents': 'Base:C(=O)([O-])[O-].[Cs+].[Cs+], amount is: 0.002219999907538295 MOLE',
         'products': '0:CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C',
         'reaction_name': '1.3.1 [N-arylation with Ar-X] Bromo Buchwald-Hartwig amination',
         'reaction_id': 'ord-56b1f4bfeebc4b8ab990b9804e798aa7',
         'conditions': "To a solution of ethyl 6-bromo-4-(2,4-difluorophenylamino)-7-ethoxyquinoline-3-carboxylate (400 mg, 0.89 mmol) and 1-(Isopropyl)piperazine (254 µl, 1.77 mmol) in dioxane was added cesium carbonate (722 mg, 2.22 mmol), tris(dibenzylideneacetone)dipalladium(0) (40.6 mg, 0.04 mmol) and rac-2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl (55.2 mg, 0.09 mmol). Reaction vessel in oil bath set to 110 °C. 11am  After 5 hours, MS shows product (major peak 499), and SM (minor peak 453).  o/n, MS shows product peak. Reaction cooled, concentrated onto silica, and purified on ISCO. 40g column, 1:1 EA:Hex, then 100% EA.  289mg yellow solid. NMR (EN00180-62-1) supports product, but some oxidised BINAP impurity (LCMS 655).  ",
         'yield': '0:65.38999938964844',
         'reference': 'https://chemrxiv.org/engage/chemrxiv/article-details/60c9e3f37792a23bfdb2d471',
         'main_product': 'CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=C(C=C(C=C3)F)F)C(=O)OCC)N4CCN(CC4)C(C)C',
         'main_product_id': '0',
         'main_product_yield': '65.38999938964844'}
        """
        query = json.loads(input_data[i])
        reactants = query['reactants']
        products = query['products']
        reaction_name = query['reaction_name']
        solvents = query["solvents"]
        solvents = cls.reformer_reaction(solvents) if solvents != "" else ""
        reactants_chain = []
        for reactant in reactants.split(";"):
            reactants_chain.append(reactant.split(",")[0].split(":")[-1])
        products = query["products"]
        products_chain = []
        for product in products.split(";"):
            products_chain.append(product.split(":")[-1])
        reaction_formula = ".".join(reactants_chain) + ">>" + ".".join(products_chain)

        catalysts = query["catalysts"]
        catalysts = cls.reformer_reaction(catalysts) if catalysts != "" else ""
        reagents = query["reagents"]
        reagents = cls.reformer_reaction(reagents) if reagents != "" else ""
        conditions = query["conditions"]
        product_yield = query["main_product_yield"]
        if reaction_name is not None and reaction_name != "":
            reaction_template = f"{reaction_name} is a chemical reaction, SMILES is sequenced-based string used to encode the molecular structure, reactants for " \
                                f"this reaction are {reactants}, SMILES for products of reactions are {products}, "
            total_reaction_template = f"{reaction_name} is a chemical reaction, SMILES is sequenced-based strings, used to encode the molecular structure, reactants for " \
                                      f"this reaction are {reactants}, SMILES for products of reactions are {products}, so the whole reaction can be represented as {reaction_formula}, "
        else:
            reaction_template = f"Considering a chemical reaction, SMILES is sequenced-based string used to encode the molecular structure, reactants for " \
                                f"this reaction are {reactants}, SMILES for products of reactions are {products}, "
            total_reaction_template = f"Considering a chemical reaction, SMILES is sequenced-based strings, used to encode the molecular structure, reactants for " \
                                      f"this reaction are {reactants}, SMILES for products of reactions are {products}, so the whole reaction can be represented as {reaction_formula}, "

        tmp = dict()
        if solvents != "":
            solvent_instruct = reaction_template + f"what solvent is used for this reaction?"
            total_reaction_template += f"solvents are {solvents}, "
            tmp["INSTRUCTION"] = solvent_instruct
            tmp["RESPONSE"] = solvents

            ord_instruct = json.dumps(
                tmp,
                ensure_ascii=False) + "\n"
            parse_ord_qa_to_instruct.write(ord_instruct)

        if reagents != "":
            reagent_instruct = reaction_template + f"what reagent is used for this reaction?"
            total_reaction_template += f"reagents are {reagents}, "
            tmp["INSTRUCTION"] = reagent_instruct
            tmp["RESPONSE"] = reagents

            ord_instruct = json.dumps(
                tmp,
                ensure_ascii=False) + "\n"
            parse_ord_qa_to_instruct.write(ord_instruct)

        if catalysts != "":
            catalyst_instruct = reaction_template + f"what catalyst is used for this reaction?"
            total_reaction_template += f"catalysts are {catalysts}, "
            tmp["INSTRUCTION"] = catalyst_instruct
            tmp["RESPONSE"] = catalysts

            ord_instruct = json.dumps(
                tmp,
                ensure_ascii=False) + "\n"
            parse_ord_qa_to_instruct.write(ord_instruct)

        if product_yield != "" and float(product_yield) != 0:
            yield_prompt = total_reaction_template + f"according to experimental procedure and condition of this " \
                                                     f"reaction: {conditions}, " \
                                                     f"please calculate the yield of products of reaction."
            tmp['INSTRUCTION'] = yield_prompt
            tmp["RESPONSE"] = product_yield
            ord_instruct = json.dumps(
                tmp,
                ensure_ascii=False) + "\n"
            parse_ord_qa_to_instruct.write(ord_instruct)

        if conditions != "":
            total_reaction_template += "please give me the experimental procedure and condition of reaction?"
            tmp['INSTRUCTION'] = total_reaction_template
            tmp["RESPONSE"] = conditions
            ord_instruct = json.dumps(
                tmp,
                ensure_ascii=False) + "\n"
            parse_ord_qa_to_instruct.write(ord_instruct)



def main():
    datasets = RawDataset("train_v2.json")
    ord_meta_data = datasets.parse_json(
        datasets.source_dir_path
    )




if __name__ == "__main__":
    main()
