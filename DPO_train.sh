export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=1

# 내가 훈련할 모델의 레포입니다.
BASIS=Bllossom/llama-3.2-Korean-Bllossom-3B

# 내가 저장한 데이터의 repo입니다
DATA=/home/nlpgpu8/hdd2/suyun/DPO_practice/llama3B_DPO/data/dataset/ko_20k

# DPO가 진행된 후 저장되는 폴더의 명칭입니다.
output_directory=/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/output_dir/data_ko

# 훈련이 끝난 어댑터와, 기존의 SFT를 merge한 이후 저장되는 폴더 이름입니다.
final_dir=/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_ko


# accelerate을 통해 분산학습을 시도하셔도 됩니다.
python training_DPO.py \
    --model_name_or_path $BASIS \
    --output_dir $output_directory \
    --datapath $DATA \
    --num_epochs 1 \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_dropout 0.01 \
    --per_device_train_batch_size 4 \
    --lr_scheduler_type 'linear' \
    --gradient_accumulation_steps 4 \
    --eval_step 100 \
    --max_prompt_length 512 \
    --max_length 512 \



# 훈련이 끝나고, 훈련된 최종 adapter와 merge를 시도합니다.
python merge.py \
    --base_model_name_or_path $BASIS \
    --peft_model_path $output_directory \
    --output_dir $final_dir \
    --device 'cuda:0'\


