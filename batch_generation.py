import os
import sys
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        CLI: bool = False,
        protein: bool = False,
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "zjunlp/llama-molinst-molecule-7b",
        prompt_template: str = "",
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
        test_data_order: int = 0,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    if protein == False:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, bos_token='<s>', eos_token='</s>', add_bos_token=True,
                                                   add_eos_token=False)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            # device_map="auto",
            device_map={"": 0}
        )
        if protein == False:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if protein == False:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if protein == False:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
            # mol-instructions

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    add_token_path = "/home/zhangyu/MolScribe/molscribe/vocab/vocab_uspto.json"
    with open(add_token_path, 'r') as fcc_file:
        add_token_dict = json.load(fcc_file)
    add_tokens = list(add_token_dict.keys())
    the_number_of_new_tokens = tokenizer.add_tokens(add_tokens)
    print(the_number_of_new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(base_model)

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            protein=False,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            repetition_penalty=1,
            max_new_tokens=128,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if protein == False:
            do_sample = False
        else:
            do_sample = True
        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        re = prompter.get_response(output)
        # remove the last ']' or ‘#’
        last_bracket_index = re.find('#')
        if last_bracket_index != -1:
            re = re[:last_bracket_index]

        last_bracket_index = re.rfind(']')
        if last_bracket_index != -1:
            re = re[:last_bracket_index + 1]

        return re

    def eva_data_loader(path):
        with open(path, 'r') as fcc_file:
            data = fcc_file.readlines()
        return data

    if CLI:
        eva_data_path = f"/home/zhangyu/data/uspto/reagents000{test_data_order}.json"
        reagent_total_num, reagent_correct = 0, 0
        product_total_num, product_correct = 0, 0
        reactant_total_num, reactant_correct = 0, 0
        preds = []
        preds_dict = defaultdict(list)
        # get instruction
        # import pudb
        # pudb.set_trace()
        test_data = eva_data_loader(eva_data_path)
        total_length = len(test_data)
        # preds = parse_multi_proc(
        #     test_data,
        #     model,
        #     prompter,
        #     tokenizer)
        for i in tqdm(range(0, total_length)):
            instruct_data = json.loads(test_data[i])
            instruction = instruct_data['instruction']
            input_text = instruct_data['input']
            output = evaluate(
                              instruction,
                              input=input_text,
                              temperature=0.1,
                              top_p=0.75,
                              top_k=40,
                              num_beams=4,
                              repetition_penalty=1,
                              max_new_tokens=256)
            preds.append(output)
        preds_dict['pred_reagents'] = preds
        print(preds)
        df = pd.DataFrame(preds_dict)
        csv_columns = ["pred_reagents"]
        df.to_csv(f"./tmp_{test_data_order}.csv", index=False, sep=",", columns=csv_columns)
        labels = [json.loads(line)['output'] for line in test_data]
        reagent_correct = acc_metric(labels, preds)
        reagent_acc = reagent_correct / total_length

        print(reagent_acc)

        # elif instruct_data['type'] == 'products':
        #
        # elif instruct_data['type'] == 'reactants':
        # elif instruct_data['type'] == 'classification':
        # elif instruct_data['type'] == 'yield':

        # print the output
        # print(output)

    else:
        mytheme = gr.themes.Default().set(
            slider_color="#0000FF",
        )
        gr.Interface(
            theme=mytheme,
            title="🧪 BAI-Chem",
            description="It is a 7B-parameter vicuna model finetuned to follow instructions. It is trained on the mol instruction dataset and makes use of the Huggingface LLaMA implementation.",
            # noqa: E501
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2,
                    label="Instruction",
                    placeholder="Please give me some details about this molecule.",
                ),
                gr.components.Textbox(
                    lines=2,
                    label="Input",
                    placeholder="[C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][=Branch1][C][=O][O][C@H1][Branch2][Ring1][=Branch1][C][O][C][=Branch1][C][=O][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][O][P][=Branch1][C][=O][Branch1][C][O][O][C][C@@H1][Branch1][=Branch1][C][=Branch1][C][=O][O][N]",
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.1, label="Temperature"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.75, label="Top p"
                ),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=40, label="Top k"
                ),
                gr.components.Slider(
                    minimum=1, maximum=4, step=1, value=4, label="Beams"
                ),
                gr.components.Slider(
                    minimum=1, maximum=5, value=1, label="Repetition penalty"
                ),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
        ).launch(server_name="0.0.0.0", share=share_gradio)




def parse_multi_proc(test_data, model, prompter, tokenizer):
    total_length = len(test_data)
    cpus = 48
    pool = Pool(cpus)
    per_pool_num = total_length // cpus
    for i in range(cpus - 1):
        locals()["process_%s" % i] = pool.apply_async(data_proc, args=(test_data, i * per_pool_num,
                                                                       i * per_pool_num + per_pool_num,
                                                                       model, prompter, tokenizer,))
    locals()["process_%s" % (i + 1)] = pool.apply_async(data_proc,
                                                        args=(
                                                            test_data, (i + 1) * per_pool_num, total_length,
                                                            model, prompter, tokenizer,))
    pool.close()
    pool.join()
    total_pred_results = []
    for i in range(cpus):
        # print(i)
        pred = locals()["process_{}".format(str(i))].get()
        total_pred_results.extend(pred)
    return total_pred_results


def data_proc(data, start_index, end_index, model, prompter, tokenizer):
    tmp_output = []
    for idx in tqdm(range(start_index, end_index)):
        instruct_data = json.loads(data[idx])
        instruction = instruct_data['instruction']
        input_text = instruct_data['input']

        output = evaluate(model,
                          prompter,
                          tokenizer,
                          instruction,
                          input=input_text,
                          temperature=0.1,
                          top_p=0.75,
                          top_k=40,
                          num_beams=4,
                          repetition_penalty=1,
                          max_new_tokens=256)

        tmp_output.append(output)
    return tmp_output


def acc_metric(labels, preds):
    correct = 0
    for (label, pred) in zip(labels, preds):
        label = label.split('.')
        pred = pred.split('</s>')[0].split('.')
        if len(pred) != len(labels):
            continue
        else:
            label.sort()
            pred.sort()
            flag = all([pred[i] == labels[i] for i in range(len(pred))])
            if flag:
                correct += 1
    return correct



if __name__ == "__main__":
    fire.Fire(main)
