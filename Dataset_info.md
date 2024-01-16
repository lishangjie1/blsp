
# Pretraining Dataset
ASR类:
1. LibriSpeech (http://www.openslr.org/12/)
2. Gigaspeech
3. Common voice

# SFT Dataset:
1. MusicCaps (https://huggingface.co/datasets/google/MusicCaps/tree/main) 来自音乐家手写
2. Music instruct (https://huggingface.co/datasets/m-a-p/Music-Instruct/tree/main/MusicInstruct) 基于MusicCaps+GPT-4构建问答对，并利用GPT-4过滤不符合事实(caption)的问答对。
3. AudioSet weak-label数据集，常用于分类(如训练AST)，也可以生成一些closed_end的问答对(openaqa所做的)
    1. Ontology.json (https://github.com/audioset/ontology)
    2. Unbalance_train_segments.csv (https://research.google.com/audioset/download.html)
    3. As_strong_train.csv (https://research.google.com/audioset/download_strong.html)
4. AudioCaps 针对audioset中的audio，人工构建的46k的caption数据集
4. OpenAQA / OpenASQA (https://github.com/YuanGongND/ltu?tab=readme-ov-file#openaqa-ltu-and-openasqa-ltu-as-dataset)
5. Vggsound (https://github.com/hche11/VGGSound?tab=readme-ov-file)
6. Llark中使用的music数据集如fma，以及llark中抽取音频节拍、弦的工具，根据多个表格数据特征，基于gpt构建问答对。

# 相关研究社区
Multimodal Art Projection (https://huggingface.co/m-a-p): MUPT, MERT, music2vec 


# llark的一次gpt instruction生成案例：

You are an expert AI assistant that is knowledgeable about music production, musical structure, music
history, and music styles, and you are hearing audio of a short clip of music. What you hear is
described in the JSON-formatted caption below, describing the same audio clip you are listening to.
Answer all questions as if you are hearing the audio clip. This caption is provided in a JSON list of
the form: [{"some_key": "some_value", "other_key": "other_value"}], where the keys and values
represent metadata about the music clip.
The JSON may contain the following fields:
'album.information': optional user-provided information about the album.
'album.tags': optional user-provided tags associated with the track album.
'artist.tags': optional user-provided tags associated with the track artist.
'track.genre_top': the top genre for the track (most frequent as determined by user votes).
'track.genres_all': all genre labels for the track.
'track.information': optional user-provided information about the track.
'track.language_code': the language of the track.
tempo_in_beats_per_minute_madmom: the tempo of the track in beats per minute (BPM).
downbeats_madmom: a list of the downbeats in the song, containing their timing ("time") and their
associated beat ("beat_number"). For example, beat_number 1 indicates the first beat of every measure
of the song. The maximum beat_number indicates the time signature (for instance, a song with
beat_number 4 will be in 4/4 time).
chords: a list of the chords of the song, containing their start time, end time, and the chord being
played.
key: the key of the song.
Design a conversation between you and a person asking about this music. The answers should be in a
tone that an AI assistant is hearing the music and answering the question. Ask diverse questions and
give corresponding answers.
Ask factual questions about the musical characteristics and content of the song, including the style
and emotions, audio characteristics, harmonic structure, presence of various instruments and vocals,
tempo, genre, relative ordering of events in the clip, etc.
Only include questions that have definite answers based on the provided metadata or your background
knowledge of this specific music as an intelligent AI assistant. Write as many question as you can
using the provided inputs. Try to include a mixture of simple questions ("Is there a saxophone in the
song?" "Are there vocals in the clip?" "What is the approximate tempo of the clip in beats per minute
(BPM)?")) and more complex questions (""How would you describe the overall mood and emotions conveyed
by the song?"). Make the questions as diverse as possible, and ask about as many different aspects of
the song as possible. Do not mention the name of the artist in the response.
Again, do not ask about uncertain details. Provide detailed answers when answering complex questions.
For example, give detailed examples or reasoning steps to make the content more convincing and
well-organized. Explain any musical concepts that would be unfamiliar to a non-musician. You can
include multiple paragraphs if necessary. Make sure that the generated questions contain questions
asking about the musical characteristics and content of the song. If there are multiple plausible
answers to a question, make sure to mention all of the plausible choices. Do not specifically
reference the provided metadata in the response; instead, respond as if you are hearing the song and
reporting facts about what you hear.
IMPORTANT: Do not use the word "metadata" anywhere in the answers to the questions. DO NOT disclose
that metadata about the song is provided to you. Always answer as if you are an expert who is
listening to the audio.
Return a single JSON list object containing the question-answer pairs. Each element in the JSON list
should be a JSON object that has the following structure: {"question": "<QUESTION TEXT GOES HERE>",
"answer": "<ANSWER TEXT GOES HERE>"}
The given json file is as follow:
{"track.id": 2, "chords": [{"start_time": 0.0, "end_time": 0.2, "chord": "no chord"}, {"start_time": 0.2, "end_time": 23.7, "chord": "Cminor"}, {"start_time": 23.7, "end_time": 26.5, "chord": "Cmajor"}, {"start_time": 26.5, "end_time": 30.0, "chord": "Cminor"}], "downbeats_madmom": [{"time": 0.08, "beat_number": 4}, {"time": 0.45, "beat_number": 1}, {"time": 0.83, "beat_number": 2}, {"time": 1.17, "beat_number": 3}, {"time": 1.55, "beat_number": 4}, {"time": 1.89, "beat_number": 1}, {"time": 2.27, "beat_number": 2}, {"time": 2.62, "beat_number": 3}, {"time": 3.0, "beat_number": 4}, {"time": 3.34, "beat_number": 1}, {"time": 3.72, "beat_number": 2}, {"time": 4.07, "beat_number": 3}, {"time": 4.42, "beat_number": 4}, {"time": 4.79, "beat_number": 1}, {"time": 5.15, "beat_number": 2}, {"time": 5.51, "beat_number": 3}, {"time": 5.87, "beat_number": 4}, {"time": 6.23, "beat_number": 1}, {"time": 6.61, "beat_number": 2}, {"time": 6.96, "beat_number": 3}, {"time": 7.32, "beat_number": 4}, {"time": 7.68, "beat_number": 1}, {"time": 8.04, "beat_number": 2}, {"time": 8.4, "beat_number": 3}, {"time": 8.77, "beat_number": 4}, {"time": 9.12, "beat_number": 1}, {"time": 9.49, "beat_number": 2}, {"time": 9.85, "beat_number": 3}, {"time": 10.2, "beat_number": 4}, {"time": 10.57, "beat_number": 1}, {"time": 10.93, "beat_number": 2}, {"time": 11.3, "beat_number": 3}, {"time": 11.65, "beat_number": 4}, {"time": 12.02, "beat_number": 1}, {"time": 12.38, "beat_number": 2}, {"time": 12.74, "beat_number": 3}, {"time": 13.1, "beat_number": 4}, {"time": 13.47, "beat_number": 1}, {"time": 13.82, "beat_number": 2}, {"time": 14.19, "beat_number": 3}, {"time": 14.55, "beat_number": 4}, {"time": 14.91, "beat_number": 1}, {"time": 15.27, "beat_number": 2}, {"time": 15.62, "beat_number": 3}, {"time": 15.99, "beat_number": 4}, {"time": 16.36, "beat_number": 1}, {"time": 16.71, "beat_number": 2}, {"time": 17.08, "beat_number": 3}, {"time": 17.44, "beat_number": 4}, {"time": 17.81, "beat_number": 1}, {"time": 18.18, "beat_number": 2}, {"time": 18.53, "beat_number": 3}, {"time": 18.89, "beat_number": 4}, {"time": 19.25, "beat_number": 1}, {"time": 19.61, "beat_number": 2}, {"time": 19.97, "beat_number": 3}, {"time": 20.33, "beat_number": 4}, {"time": 20.69, "beat_number": 1}, {"time": 21.06, "beat_number": 2}, {"time": 21.42, "beat_number": 3}, {"time": 21.78, "beat_number": 4}, {"time": 22.14, "beat_number": 1}, {"time": 22.5, "beat_number": 2}, {"time": 22.87, "beat_number": 3}, {"time": 23.22, "beat_number": 4}, {"time": 23.59, "beat_number": 1}, {"time": 23.95, "beat_number": 2}, {"time": 24.31, "beat_number": 3}, {"time": 24.67, "beat_number": 4}, {"time": 25.03, "beat_number": 1}, {"time": 25.4, "beat_number": 2}, {"time": 25.76, "beat_number": 3}, {"time": 26.12, "beat_number": 4}, {"time": 26.48, "beat_number": 1}, {"time": 26.85, "beat_number": 2}, {"time": 27.2, "beat_number": 3}, {"time": 27.57, "beat_number": 4}, {"time": 27.92, "beat_number": 1}, {"time": 28.28, "beat_number": 2}, {"time": 28.65, "beat_number": 3}, {"time": 29.0, "beat_number": 4}, {"time": 29.37, "beat_number": 1}, {"time": 29.74, "beat_number": 2}], "key": "C minor", "tempo_in_beats_per_minute_madmom": 83.3, "album.information": "", "album.tags": "[]", "artist.tags": "['awol']", "track.genre_top": "Hip-Hop", "track.genres_all": ["Hip-Hop"], "track.information": "nan", "track.language_code": "English"}


