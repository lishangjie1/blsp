import os
import argparse
import json
from tqdm import tqdm

from transformers import LlamaTokenizer, WhisperFeatureExtractor
from transformers import GenerationConfig
from blsp.src.modeling_blsp import BlspModel
from blsp.src.configuration_blsp import BlspConfig
from blsp.src.speech_text_paired_dataset import get_waveform
from blsp.src.ast_feature_extractor import ASTFeatureExtractor
generation_config = GenerationConfig(
    max_new_tokens=128,
    min_new_tokens=1,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    num_beams=1,
    num_return_sequences=1,
)
GLOBAL_MODEL = None
GLOBAL_EXTRACTOR = None
GLOBAL_TOKENIZER = None
recent_audio_file = None
audio_name = None
DEFAULT_CONVERSATION_HEADER = None
def Load_Model():
    global GLOBAL_MODEL, GLOBAL_EXTRACTOR, GLOBAL_TOKENIZER, generation_config, DEFAULT_CONVERSATION_HEADER
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="Path to the deepspeed_config"
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
    parser.add_argument(
		"--local-rank",
		type=int,
	)
    args = parser.parse_args()

    from blsp.src.special_tokens import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN
    from blsp.src import conversation as conversation_lib
    DEFAULT_CONVERSATION_HEADER = f"{conversation_lib.default_conversation.system}"


    tokenizer = LlamaTokenizer.from_pretrained(args.blsp_model)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = 0 # llama do not have pad_token_id, need to add manually
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")

    # if DEFAULT_AUDIO_START_TOKEN not in tokenizer.get_vocab():
    #     num_new_tokens = tokenizer.add_tokens(
    #             [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN],
    #             special_tokens=True,
    #         )
    
    if args.audio_model_type == "whisper":
        EXTRACTOR_CLASS = WhisperFeatureExtractor
    elif args.audio_model_type == "ast":
        EXTRACTOR_CLASS = ASTFeatureExtractor
    else:
        raise Exception("Unknown audio model type")
    extractor = EXTRACTOR_CLASS.from_pretrained(args.blsp_model)
    model = BlspModel.from_pretrained(args.blsp_model) # may need to change vocab_size in config of model manually (e.g., from 32000 to 32002)
    model = model.cuda()
    model.eval()

    GLOBAL_MODEL = model
    GLOBAL_EXTRACTOR = extractor
    GLOBAL_TOKENIZER = tokenizer

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

    


            
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template(f'upload.html')

@app.route('/file_upload', methods=['POST'])
def file_upload():
    global recent_audio_file, audio_name
    file = request.files.get('file')
    if file:
        save_path = 'uploads/' + file.filename
        file.save(save_path)
        recent_audio_file = save_path
        audio_name = file.filename
        return render_template(f'upload.html',prediction_display_area=f'当前文件: {file.filename}')
    else:
        return render_template(f'upload.html',prediction_display_area='上传失败')


@app.route('/predict', methods=['POST'])
def predict():
    
    audio = recent_audio_file
    if audio is not None:
        speech = get_waveform(audio, output_sample_rate=GLOBAL_EXTRACTOR.sampling_rate)
        speech_inputs = GLOBAL_EXTRACTOR(
            speech,
            sampling_rate=GLOBAL_EXTRACTOR.sampling_rate,
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
    else:
        speech_values = None
        speech_attention_mask = None

    # interactive generation
    
    question = request.form["message"].strip()
    
    instruction = f"Question: {question}"
    input_str = f"{DEFAULT_CONVERSATION_HEADER}\n\n###[Human]:{instruction}" + "\n\n###[Audio]"
    input_ids = GLOBAL_TOKENIZER(input_str, return_tensors="pt").input_ids.cuda()
    
    # A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n###[Human]:who are you?\n\n###[Assistant]:
    suffix_input_str = "\n\n###[Assistant]:" #f"{DEFAULT_AUDIO_END_TOKEN}" + "\n\n\n###[Assistant]:"
    suffix_input_ids = GLOBAL_TOKENIZER(suffix_input_str, return_tensors="pt").input_ids[:,1:].cuda()

    output = GLOBAL_MODEL.generate(
        input_ids=input_ids,
        suffix_input_ids=suffix_input_ids,
        speech_values=speech_values,
        speech_attention_mask=speech_attention_mask,
        generation_config=generation_config,
    )
    response = GLOBAL_TOKENIZER.decode(output[0], skip_special_tokens=True)
    answer = response.split('\n')[0]
    return render_template(f'upload.html',prediction_display_area=f'当前文件: {audio_name}', question_area=f'问题: {question}', answer_area=f'回答: {answer}')
    
if __name__ == "__main__":
    Load_Model()
    print("http://0.0.0.0:5010")
    app.run(host='0.0.0.0',port=5010)
