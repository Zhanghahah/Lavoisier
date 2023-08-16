#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#python prompt_eval.py \
#    --model_name_or_path_baseline facebook/opt-6.7b \
#    --model_name_or_path_finetune yuyuc/chem-upsto


#python3 prompt_eval.py \
#    --model_name_or_path_baseline facebook/opt-6.7b \
#    --model_name_or_path_finetune yuyuc/chem-upsto

python3 prompt_eval_single.py \
--model_name_or_path_baseline /home/zhangyu/llama-7b-hf \
--model_name_or_path_finetune /home/zhangyu/llama-7b-instruct-base-chem



