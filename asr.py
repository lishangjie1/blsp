import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, AutoProcessor, pipeline, AutoTokenizer
from datasets import load_dataset
from blsp.src.speech_text_paired_dataset import get_waveform

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "/mnt/nas/users/lsj/music/models/whisper-large-v2"
extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.to(device)

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

generation_config = model.generation_config

# generation_config
# task: {"transcribe", "translate"}
# timestamp
# language

generation_config.task = "transcribe"
generation_config.decoder_start_token_id = 50258 # 50257:<|endoftext|>, 50258:<|startoftranscript|>,50259:<|en|>
language = "en"
result = model.generate(**speech_inputs, generation_config=generation_config, language=language, return_timestamps=False)
result_str = tokenizer.decode(result[0])
print(result_str)

