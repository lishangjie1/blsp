
set -e

model_path="/mnt/nas/users/lsj/music/models/llama2_7b_chat_ast_lora/epoch0_iter2500_loss0.938"
# pretrain_model_path="/mnt/nas/users/lsj/llm/pretrained_model/llama_7b"
# audio_encoder_path="/mnt/nas/users/lsj/music/models/whisper-large-v2"
# audio_encoder_path="/mnt/nas/users/lsj/music/models/AST"
# copy tokenizer
# cp $pretrain_model_path/tokenizer* $model_path/
# copy preprocessor_config
# cp $audio_encoder_path/preprocessor_config.json $model_path/
# change vocab_size in config.json

python blsp/generate.py \
    --input_file /mnt/nas/users/lsj/music/blsp/sft_data_openaqa/test_instruct.json \
    --output_file generate_result.json \
    --blsp_model $model_path \
    --audio_model_type "ast"
