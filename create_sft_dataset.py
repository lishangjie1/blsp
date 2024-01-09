import pandas as pd
import jsonlines
import os
# download youtube video/audio by music_caps_dl (https://github.com/seungheondoh/music_caps_dl.git)
music_caps = pd.read_csv("musiccaps-public.csv") # download from MusicCaps (https://huggingface.co/datasets/google/MusicCaps)
instructs = jsonlines.open("music_instruct_short.json").read()["QA"] # download from Music-Instruct (https://huggingface.co/datasets/m-a-p/Music-Instruct)
raw_audio_dir = "raw_audio"
target_audio_dir = "/mnt/nas/users/lsj/music/blsp/sft_data/audio"
print("information of musiccaps")
print(len(music_caps))
print(music_caps.columns)


print("information of music instruction")
print(len(instructs))

columns = music_caps.columns

music_caps_dic = {}
for i in range(len(music_caps)):
    ytid = music_caps.iloc[i, 0]
    caption = music_caps.iloc[i, 5]

    assert ytid not in music_caps_dic
    music_caps_dic[ytid] = {}
    music_caps_dic[ytid]["cap"] = caption


raw_audios = {}

for audio_name in os.listdir(raw_audio_dir):
    prefix = audio_name.split("-")[0]
    prefix = prefix[1:-1]
    raw_audios[prefix] = audio_name

skip_num = 0
with jsonlines.open("sft_data/instruct.json", "w") as f: 
    for i in range(len(instructs)):
        instruct = instructs[i]
        ytid = instruct["ytid"]

        

        if ytid in music_caps_dic and ytid in raw_audios:
            audio_name = raw_audios[ytid]
            start_time, end_time = audio_name.split('.')[0].split('-')[1:]
            start_time, end_time = start_time.strip('['), end_time.strip(']')

            # transform raw audio (webm/m4a) into .wav
            if not os.path.exists(f"{target_audio_dir}/{ytid}.wav"):
                os.system(f"ffmpeg -i {raw_audio_dir}/{audio_name} -ss {start_time} -to {end_time} {target_audio_dir}/{ytid}.wav")
            sample = {}

            mc_caption = music_caps_dic[ytid]["cap"]
            mi_caption = instruct["caption"]

            # assert mi_caption == mc_caption, f"{mi_caption} || {mc_caption}"

            sample["audio"] = f"{target_audio_dir}/{ytid}.wav"
            sample["question"] = instruct["question"]
            sample["answer"] = instruct["answer"]
            sample["caption"] = instruct["caption"]

            f.write(sample)
        else:
            skip_num += 1
            print(f"Skip {ytid}, skip_num: {skip_num}")



    


