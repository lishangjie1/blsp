import json
import pandas as pd
import os
import jsonlines
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import time
import sys
import soundfile as sf
# need to download csv or json files:
# music: music_instruct_short.json musiccaps-public.csv
# Audio_set: ontology.json as_strong_train.json unbalanced_train_segments.csv
# Openaqa: openaqa_5.6M.json vggsound.csv
def _download_audio(x):
    (
        ytid,
        start_time,
        end_time,
        save_path,
    ) = x
    timeout = 20
    pid = os.getpid()
    
    # download audio
    audio_name = f"{save_path}/{ytid}-[{start_time}-{end_time}]"
    # print(pid, audio_name)
    if not os.path.exists(f"{audio_name}.wav"):
        try:

            cmd = f"bash download_single.sh {ytid} {start_time} {end_time} {audio_name} wav"
            p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, shell=True)
            t_beginning = time.time()
            while True:
                if p.poll() is not None:
                    if p.returncode != 0:
                        # print(p.returncode)
                        sys.exit(9)
                    break
                seconds_passed = time.time() - t_beginning
                if seconds_passed > timeout:
                    p.kill()
                    print(f"Timeout! Command will be kill. audio_name: {audio_name}")
                    break
                time.sleep(1)
        except:
            pass

    

def download_ps(ytid, st_list, ed_list, save_path, num_processes=None, desc=None):
    with mp.Pool(processes=num_processes) as pool, tqdm(total=len(ytid), desc=desc) as pbar:

            for _ in tqdm(
                pool.imap(
                    _download_audio,
                    zip(ytid, st_list, ed_list, [save_path] * len(ytid)),
                ),
                total=len(ytid),
            ):
                pbar.update()


def get_data():
    # extract qa pair for `Audioset` `Audiocaps` `VGGSOUND` 
    # skip `FreeSound` `FSD50K` `clotho_v2`

    # `Audioset` : 2M dataset with only weak-label for sound class, youtube video ytid/start_time/end_time refer to "unbalanced_train_segments.csv"
    # `Audiocaps` : 46K dataset with human-written captions on Audioset, ytid/start_time/end_time refer to "unbalanced_train_segments.csv"
    # VGGSOUND: 200K dataset with weak-label (created by computer-vision tecnology), ytid/start_time refer to "vggsound.csv"

    # dataset type
    # "as_2m": only weak-label, can be used to create the following tasks: (1) cla_label_des, 
    # "as_20k"
    # "as_strong_train": 9M sound events strong-label (multiple sound events and precise timestamp), can be used to create the following tasks: (1) open-ended question (2) temporal_single (3) temporal_order (4) temporal (5) cla_label

    # "audiocaps_train": human-written caption, can be used to create the following tasks: (1) open-ended question

    # "vggsound_train": weak-label (created by computer-vision tecnology), can be used to create the following tasks: (1) open-ended question (2) cla_label


    # load unbalanced_train_segments.csv
    unbalanced_train_segments = pd.read_csv("unbalanced_train_segments.csv", sep=', ')
    map_as = {}
    for i in range(len(unbalanced_train_segments)):
        ytid = unbalanced_train_segments.iloc[i, 0]
        start_time, end_time = unbalanced_train_segments.iloc[i, 1], unbalanced_train_segments.iloc[i, 2]
        map_as[ytid] = (start_time, end_time)

    # load vggsound.csv
    vggsound = pd.read_csv("vggsound.csv", header=None)
    map_vgg ={}
    for i in range(len(vggsound)):
        ytid = vggsound.iloc[i, 0]
        start_time = vggsound.iloc[i, 1]
        sample_type = vggsound.iloc[i, 3]
        map_vgg[ytid] = (start_time, sample_type)

    cnt = 0
    skip = 0
    ytids, st_list, ed_list = [], [], []
    questions, answers = [], []
    with open("openaqa_5.6M.json") as f:
        openaqa = json.load(f)

        print(len(openaqa))

        for line in openaqa:

            if "/data/sls/audioset" in line["audio_id"]:
                question = line["instruction"]
                answer = line["output"]
                ytid = line["audio_id"].split('/')[-1].split('.')[0]
                if ytid not in map_as:
                    skip += 1
                    continue
                start_time, end_time = map_as[ytid]
                start_time, end_time = int(start_time), int(end_time)
                

            elif "vggsound" in line["audio_id"]:
                question = line["instruction"]
                answer = line["output"]
                ytid = line["audio_id"].split('/')[-1].split('.')[0]
                # drop time suffix
                ytid = '_'.join(ytid.split('_')[:-1])
                if ytid not in map_vgg:
                    skip += 1
                    continue
                start_time, sample_type = map_vgg[ytid]

                if sample_type != "train":
                    skip += 1
                    continue
                
                start_time = int(start_time)
                end_time = start_time + 10

            else:
                continue
                
            task = line["task"]
            if task == "open-ended question":
                continue

            ytids.append(ytid)
            st_list.append(start_time)
            ed_list.append(end_time)
            questions.append(question)
            answers.append(answer)

    return ytids, st_list, ed_list, questions, answers


if __name__ == "__main__":

    ytids, st_list, ed_list, questions, answers = get_data()
    print(f"Target video number: {len(ytids)}")

    # Starting Downloading
    save_path = "sft_data_openaqa/audio"

    chunk = 1000
    chunk_size = len(ytids) // chunk

    # for i in range(50):
    #     left, right = i * chunk_size, min((i+1)*chunk_size, len(ytids))

    #     download_ps(ytids[left:right], st_list[left:right], ed_list[left:right], save_path, num_processes=16)        



    with jsonlines.open("sft_data_openaqa/train_instruct.json", "w") as fw:

        cnt = 0
        total_num = len(ytids)
        for ytid, start_time, end_time, question, answer in zip(ytids, st_list, ed_list, questions, answers):
            cnt += 1
            if cnt % 1000 == 0:
                print(f"{cnt} / {total_num}")
            audio_dir = "sft_data_openaqa/audio"
            audio_name = f"{ytid}-[{start_time}-{end_time}]"
            if os.path.exists(f"{audio_dir}/{audio_name}.wav"): # downloaded successfully
                try:
                    info = sf.info(f"{audio_dir}/{audio_name}.wav")
                    is_readable = True
                except:
                    is_readable = False
                if is_readable:
                    sample = {"audio": f"{audio_name}.wav", "question": question, "answer": answer}
                    fw.write(sample)
