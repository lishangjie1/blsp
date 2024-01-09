

ytid="$1"
url=https://www.youtube.com/watch?v=$ytid
start_time=$2
end_time=$3
out_name="$4"
ext="$5"
# pip install yt-dlp
# download ffmpeg xx.7z from website and run it.
# ffmpeg_i 在下载之前通过输入关键帧进行搜索，因此容易切出额外的audio，但可以实现只下载部分audio
# 当前模式是通过输出搜索，能够准确查找start_time和end_time,但有时候会卡住(?)
# 
yt-dlp -q --no-warnings -x --audio-format $ext "https://www.youtube.com/watch?v=$ytid" -o "$out_name.$ext" --downloader ffmpeg --downloader-args "-ss $start_time -to $end_time"
