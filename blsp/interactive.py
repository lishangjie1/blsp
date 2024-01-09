import os
import argparse
import json
from tqdm import tqdm

from transformers import LlamaTokenizer, WhisperFeatureExtractor
from transformers import GenerationConfig
from src.modeling_blsp import BlspModel
from src.configuration_blsp import BlspConfig
from src.speech_text_paired_dataset import get_waveform
from src.ast_feature_extractor import ASTFeatureExtractor
generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    num_beams=1,
    num_return_sequences=1,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--audio_model_type", type=str, default="ast",
        help="type of the audio model of the blsp model", required=True
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
    args = parser.parse_args()

    from src.special_tokens import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN
    from src import conversation as conversation_lib
    DEFAULT_CONVERSATION_HEADER = f"{conversation_lib.default_conversation.system}"


    tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = 0 # llama do not have pad_token_id, need to add manually
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")

    if DEFAULT_AUDIO_START_TOKEN not in tokenizer.get_vocab():
        num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN],
                special_tokens=True,
            )
    
    if args.audio_model_type == "whisper":
        EXTRACTOR_CLASS = WhisperFeatureExtractor
    elif args.audio_model_type == "ast":
        EXTRACTOR_CLASS = ASTFeatureExtractor
    else:
        raise Exception("Unknown audio model type")
    extractor = EXTRACTOR_CLASS.from_pretrained(args.blsp_model)
    model = BlspModel.from_pretrained(args.blsp_model,force_download=True) # may need to change vocab_size in config of model manually (e.g., from 32000 to 32002)

    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )
    print(generation_config)
    model = model.cuda()
    model.eval()

    audio = args.audio_file
    speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate)
    speech_inputs = extractor(
        speech,
        sampling_rate=extractor.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt"
    )
    if "input_features" in speech_inputs:
        speech_values = speech_inputs.input_features
    elif "input_values" in speech_inputs:
        speech_values = speech_inputs.input_values
    else:
        raise Exception("No input in speech_inputs")
    speech_values = speech_values.cuda()
    speech_attention_mask = getattr(speech_inputs, "attention_mask", None)
    if speech_attention_mask is not None:
        speech_attention_mask = speech_attention_mask.cuda()

    # interactive generation
    cnt = 0
    while True:
        cnt += 1
        instruction = input(f"Question {cnt}: ").strip()
        input_str = f"{DEFAULT_CONVERSATION_HEADER}\n\n###[Human]:{instruction}\n\n" + f"{DEFAULT_AUDIO_START_TOKEN}"
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.cuda()
        

        suffix_input_str = f"{DEFAULT_AUDIO_END_TOKEN}" + "\n\n\n###[Assistant]:"
        suffix_input_ids = tokenizer(suffix_input_str, return_tensors="pt").input_ids[:,1:].cuda()

        output = model.generate(
            input_ids=input_ids,
            suffix_input_ids=suffix_input_ids,
            speech_values=speech_values,
            speech_attention_mask=speech_attention_mask,
            generation_config=generation_config,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response {cnt}: {response}\n")


            

if __name__ == "__main__":
    main()
