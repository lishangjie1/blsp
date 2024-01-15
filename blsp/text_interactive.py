import os
import argparse
import json
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, WhisperFeatureExtractor
from src.tokenization_qwen import QWenTokenizer
from src.configuration_qwen import QWenConfig
from transformers import GenerationConfig 
from src.modeling_blsp import BlspModel
from src.configuration_blsp import BlspConfig
from src.speech_text_paired_dataset import get_waveform
from src.ast_feature_extractor import ASTFeatureExtractor
from transformers import LlamaForCausalLM
from src.modeling_qwen import QWenLMHeadModel
import deepspeed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="Path to the deepspeed_config", required=True
    )
    
    parser.add_argument(
        "--instruction", type=str, default="",
        help="the general instruction for each example"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="whether do sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="top_p for generation"
    )
    parser.add_argument(
		"--local-rank",
		type=int,
	)
    args = parser.parse_args()
    #torch.cuda.set_device(args.local_rank)
    #deepspeed.init_distributed()

    from src import conversation as conversation_lib
    DEFAULT_CONVERSATION_HEADER = f"{conversation_lib.default_conversation.system}"

    
    tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
    #tokenizer = QWenTokenizer.from_pretrained(args.blsp_model)
    #model = BlspModel.from_pretrained(args.blsp_model,force_download=True) # may need to change vocab_size in config of model manually (e.g., from 32000 to 32002)
    
    model = LlamaForCausalLM.from_pretrained(args.blsp_model)
    # config = QWenConfig.from_pretrained(args.blsp_model)
    # config.fp32 = True
    # config.fp16 = False
    # config.bf16 = False
    # config.use_flash_attn = False
    # model = QWenLMHeadModel.from_pretrained(args.blsp_model,
    #                                 from_tf=bool(".ckpt" in args.blsp_model),
    #                                 config=config)
    # generation_kwargs = {
    #         "early_stopping": True,
    #         "max_new_tokens": args.max_new_tokens,
    #         "pad_token_id": tokenizer.eod_id,
    #         "bos_token_id": tokenizer.bos_token_id,
    #         "eos_token_id": tokenizer.eod_id#tokenizer.eos_token_id
    #     }
    
    generation_kwargs = {"early_stopping": True}
    # 当pad和eos相同时，transformers默认不生成pad的attention mask，通过is_pad_token_not_equal_to_eos_token_id，这里做了修改 transformers/generation/utils.py 609行
    # qwen的attention代码中仅针对casual mask做了掩码，没有对pad的attention进行掩码，这里做了修改 modeling_qwen.py 255行
    pad_token_id = None
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        pad_token_id = tokenizer.eod_id
    if pad_token_id:
        generation_kwargs["pad_token_id"] = pad_token_id

    eos_token_id = None
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        eos_token_id = tokenizer.eod_id
    if eos_token_id:
        generation_kwargs["eos_token_id"] = eos_token_id

    print(f"generation kwargs: {generation_kwargs}")
    model = model.cuda()
    model.eval()
    #model = deepspeed.init_inference(model=model, config=args.deepspeed_config)
    # interactive generation
    cnt = 0
    while True:
        cnt += 1
        instruction = input(f"Question {cnt}: ")
        input_str = f"{instruction}"
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()

        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            **generation_kwargs,
        )
        prefix_str = tokenizer.decode(input_ids[0])
        response = tokenizer.decode(output[0])
        answer = response[len(prefix_str):].split('\n')[0].strip() # drop prefix
        print(f"Response {cnt}: {response}\n")


            

if __name__ == "__main__":
    main()
