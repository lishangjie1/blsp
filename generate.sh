
set -e


python blsp/generate.py \
    --input_file train.jsonl \
    --output_file generate_result.json \
    --blsp_model /mnt/nas/users/lsj/music/models
