


llama_path=/mnt/nas/users/lsj/llm/pretrained_model/Llama-2-7b-chat-hf
audio_model_path="/mnt/nas/users/lsj/music/models/AST"
audio_model_type="ast"
# blsp_path="/mnt/nas/users/lsj/music/models/llama_7b_sft_closed"
DATA_ROOT=/mnt/nas/users/lsj/music/blsp/sft_data_openaqa
SAVE_ROOT=/mnt/nas/users/lsj/music/models/llama2_7b_chat_ast_lora_s2_max100_only_open

mkdir -p $SAVE_ROOT

echo $WORLD_SIZE # node num in DLC, need to multiply nproc
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK

# WORLD_SIZE=1
# RANK=0
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=12346

# stage 1: only train projection (subsampler+adapter, 10M parameters), closed question, lr=5e-4, epoch=2
# stage 2: all train (audio encoder, projection, lora(lora_dim=8), 105M parameters), all data, lr=2e-4, epoch=2

stage_one_model="/mnt/nas/users/lsj/music/models/llama2_7b_chat_ast_lora_s1"

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use_env \
    blsp/train_stage2.py \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "train_instruct_opened.json,train_instruct_asr.json" \
    --instruction "" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  False \
    --fp16 True \
    \
    --learning_rate 2e-4 \
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
    --offload True \
    --blsp_model "$stage_one_model"
