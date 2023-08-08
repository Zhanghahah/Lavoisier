BASE_MODEL_PATH=" "
FINETUNED_MODEL_PATH=" "

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --CLI False\
    --protein False\
    --load_8bit \
    --base_model '/home/zhangyu/public_models/vicuna-7b-hf' \
    --share_gradio True\
    --lora_weights './lora-vicuna-7b-chem-instructions-50w-07-25/' \


# yuyuc/mol-instructions-describe-lora
# zjunlp/llama-molinst-molecule-7b