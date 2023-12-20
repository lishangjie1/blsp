


llama_path=/mnt/nas/users/lsj/llm/pretrained_model/llama_7b
whisper_path="mnt/nas/users/lsj/llm/pretrained_model/whisper"

DATA_ROOT=/mnt/nas/users/lsj/music/data
SAVE_ROOT=/mnt/nas/users/lsj/music/models

mkdir -p $SAVE_ROOT

python -m torch.distributed.run --nproc_per_node=2 blsp/train_stage2.py \
    --deepspeed blsp/config/dp_config_zero1.json \
    --data $DATA_ROOT \
    --output_dir ${SAVE_ROOT} \
    --manifest_files "train.jsonl" \
    --instruction "" \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  False \
    --fp16 True \
    \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    \
    --llama_model $llama_path \
    --whisper_model $whisper_path \
    \
    --disable_tqdm True \
    \
    --logging_steps 1 \
    --save_steps 200 \
    --save_total_limit 1
