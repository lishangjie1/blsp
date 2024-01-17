import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, ASTFeatureExtractor
from blsp.src.modeling_whisper_encoder import WhisperEncoder
from blsp.src.modeling_ast_encoder import ASTModel
from datasets import load_dataset
from blsp.src.speech_text_paired_dataset import get_waveform

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "/mnt/nas/users/lsj/music/models/whisper-large-v2"
extractor = WhisperFeatureExtractor.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = WhisperForConditionalGeneration.from_pretrained(model_id)
# model.to(device)

encoder = WhisperEncoder.from_pretrained(model_id)
encoder = encoder.to(device)
audio = "8hAgH_ZJJgA-[560-570].wav"
speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate)
speech_inputs = extractor(
    speech,
    sampling_rate=extractor.sampling_rate,
    return_attention_mask=True,
    return_tensors="pt"
)


print(speech_inputs.keys())
for key in speech_inputs:
    speech_inputs[key] = speech_inputs[key].to(device)
    print(f"Encoder input: key: {key}, shape: {speech_inputs[key].shape}")

output = encoder(**speech_inputs)
print(f"Whisper output: {output.last_hidden_state.shape}")
# generation_config = model.generation_config

# generation_config
# task: {"transcribe", "translate"}
# timestamp
# language

# generation_config.task = "transcribe"
# generation_config.decoder_start_token_id = 50258 # 50257:<|endoftext|>, 50258:<|startoftranscript|>,50259:<|en|>
# language = "en"
# result = model.generate(**speech_inputs, generation_config=generation_config, language=language, return_timestamps=False)
# result_str = tokenizer.decode(result[0])
# print(result_str)



#### ast feature extractor example
model_id = "/mnt/nas/users/lsj/music/models/AST"
extractor = ASTFeatureExtractor.from_pretrained(model_id)
encoder = ASTModel.from_pretrained(model_id)
encoder = encoder.to(device)
audio = "8hAgH_ZJJgA-[560-570].wav"
speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate)
speech_inputs = extractor(
    speech,
    sampling_rate=extractor.sampling_rate,
    return_attention_mask=True,
    return_tensors="pt"
)
w2v_args = {
            "input_features": speech_inputs.input_values,
            "attention_mask": None,
        }

print(w2v_args.keys())
for key in w2v_args:
    if w2v_args[key] is not None:
        w2v_args[key] = w2v_args[key].to(device)
        print(f"Encoder input: key: {key}, shape: {w2v_args[key].shape}")
output = encoder(**w2v_args)
print(f"AST output: {output.last_hidden_state.shape}")

