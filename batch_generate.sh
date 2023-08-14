BASE_MODEL_PATH=" "
FINETUNED_MODEL_PATH=" "

for i in {0..7};
#echo "first parameter conducted: $1";
#echo "second parameter conducted: $2";
do
CUDA_VISIBLE_DEVICES=$i python batch_generation.py \
    --CLI True\
    --protein False\
    --load_8bit \
    --base_model '/home/zhangyu/public_models/vicuna-7b-hf' \
    --share_gradio True\
    --lora_weights './lora-vicuna-7b-chem-instructions-uspto-500-mt-50w-08-08/' \
    --test_data_order "$i"
done

