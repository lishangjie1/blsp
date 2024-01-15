
set -e

model_path="/mnt/nas/users/lsj/music/models/llama2_7b_chat_ast_lora_s2_max100_only_open"
# pretrain_model_path="/mnt/nas/users/lsj/llm/pretrained_model/llama_7b"
# audio_encoder_path="/mnt/nas/users/lsj/music/models/whisper-large-v2"
# audio_encoder_path="/mnt/nas/users/lsj/music/models/AST"
# copy tokenizer
# cp $pretrain_model_path/tokenizer* $model_path/
# copy preprocessor_config
# cp $audio_encoder_path/preprocessor_config.json $model_path/
# change vocab_size in config.json
export CUDA_VISIBLE_DEVICES=1
python blsp/interactive.py \
    --audio_file /mnt/nas/users/lsj/music/blsp/00M9FhCet6s.wav \
    --blsp_model $model_path \
    --audio_model_type "ast"
