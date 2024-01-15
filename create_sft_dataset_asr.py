import os
import soundfile as sf
import jsonlines
import numpy as np
asr_data_dir = "/mnt/nas/users/lsj/music/blsp/asr_data"
Libri_path = f"{asr_data_dir}/LibriSpeech/train-clean-100"
template_file = f"{asr_data_dir}/template"
templates = []
with open(template_file) as f:
    for line in f:
        templates.append(line.strip())



with jsonlines.open(f"{asr_data_dir}/train_instruct_asr.json", "w") as fw:
    
    for d1 in os.listdir(Libri_path):
        d1_path = os.path.join(Libri_path, d1)
        if not os.path.isdir(d1_path):
            continue

        for d2 in os.listdir(d1_path):
            d2_path = os.path.join(d1_path, d2)
            if not os.path.isdir(d2_path):
                continue
            trans_txt = f"{d2_path}/{d1}-{d2}.trans.txt"

            with open(trans_txt) as f:
                for line in f:
                    audio_name, content = line.strip().split(' ', 1)

                    audio_path = os.path.join(d2_path, audio_name+".flac")
                    content = content.lower().capitalize()

                    try:
                        sf.info(audio_path)
                        is_readable = True
                    except:
                        is_readable = False

                    assert is_readable


                    random_choice_idx = np.random.choice(len(templates))
                    sample = {}
                    sample["audio"] = audio_path
                    sample["question"] = templates[random_choice_idx]
                    sample["answer"] = content 
                    fw.write(sample)                   

            




