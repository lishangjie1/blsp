


llama_path=/mnt/nas/users/lsj/llm/pretrained_model/llama_7b
audio_model_path="/mnt/nas/users/lsj/music/models/AST"
audio_model_type="ast"
# blsp_path="/mnt/nas/users/lsj/music/models/llama_7b_sft_closed"
DATA_ROOT=/mnt/nas/users/lsj/music/blsp/sft_data_openaqa
SAVE_ROOT=/mnt/nas/users/lsj/music/models/llama_7b_sft_ast_lora

mkdir -p $SAVE_ROOT

echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK

# WORLD_SIZE=1
# RANK=0
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=12346

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use_env \
    blsp/train_stage2.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "train_instruct.json" \
    --instruction "" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  False \
    --fp16 True \
    \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    \
    --llama_model $llama_path \
    --audio_model_type "$audio_model_type" \
    --audio_model $audio_model_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 10 \
    --save_steps 2500 \
    --save_total_limit 10 \
    --lora_dim 8 \
    --offload True 
