

import os
import sys
import jsonlines
import numpy as np
QA_RESPONSE_DIR = sys.argv[1]
AUDIO_ENCODING_DIR = sys.argv[2]
OUTPUT = sys.argv[3]
ext = sys.argv[4]

dataset_keys = ["audio", "question", "answer"]

with jsonlines.open(OUTPUT, "w") as fw:
    
    for fname in os.listdir(QA_RESPONSE_DIR):
        sample = {}
        track_id = fname.split('.')[0]
        audio_path = os.path.join(AUDIO_ENCODING_DIR, f"{track_id}.{ext}")

        response_file_path = os.path.join(QA_RESPONSE_DIR, f"{track_id}.jsonl") 

        with jsonlines.open(response_file_path) as f:
            for response in f:
                question, answer = response["question"], response["answer"]
                sample = {"audio": audio_path, "question": question, "answer": answer}
        
                fw.write(sample)
  

# result examples
# python merge_speech_to_text_json.py /mnt/nas/users/lsj/music/data/fma_qa_result /mnt/nas/users/lsj/music/data/fma_small/000 /mnt/nas/users/lsj/music/data/train.jsonl mp3
# {"audio": "data/fma_small/000/000002.mp3", "question": "What is the key of the song?", "answer": "The key of the song is C minor."}
# {"audio": "data/fma_small/000/000002.mp3", "question": "What is the tempo of the song in beats per minute (BPM)?", "answer": "The tempo of the song is approximately 83.3 BPM."}
# {"audio": "data/fma_small/000/000002.mp3", "question": "Are there vocals in the clip?", "answer": "No, there are no vocals in the clip."}
# {"audio": "data/fma_small/000/000002.mp3", "question": "What are the chords played in the song?", "answer": "The chords played in the song are C minor, C major, and C minor."}








