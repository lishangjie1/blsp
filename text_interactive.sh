
set -e

model_path="/mnt/nas/users/lsj/llm/pretrained_model/Llama-2-7b-chat-hf"

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    blsp/text_interactive.py \
    --blsp_model $model_path \
    --deepspeed_config "ds_config_inference.json"
