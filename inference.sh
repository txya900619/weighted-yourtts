# tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx VCTK_p374

# 宣告兩個 list
text_list=("'que4', 'ding4', 'wu2', 'yi2', 'shi4', 'ge5', 'mei2', 'jing1', 'guo4', 'ren4', 'he2', 'gui1', 'hua4', 'shen3', 'pi1', 'de5', 'wei2', 'zhang1', 'jian4', 'zhu2'" "she4', 'ji4', 'chu1', 'lai2', 'de5', 'jia1', 'ju4', 'nai4', 'shou4', 'li4', 'ye3', 'cha1'" "'que4', 'ding4', 'wu2', 'yi2', 'shi4', 'ge5', 'mei2', 'jing1', 'guo4', 'ren4', 'he2', 'gui1', 'hua4', 'shen3', 'pi1', 'de5', 'wei2', 'zhang1', 'jian4', 'zhu2'" "she4', 'ji4', 'chu1', 'lai2', 'de5', 'jia1', 'ju4', 'nai4', 'shou4', 'li4', 'ye3', 'cha1'")
speaker_idx_list=("tE_a4prlOSk_1253102-1254269" "pnUnvpYkQeU_0567502-0569502" "tE_a4prlOSk_1253102-1254269" "pnUnvpYkQeU_0567502-0569502")

# 宣告變數
out_path="/mnt/md1/user_wago/TTS/recipes/mandarin_drama/yourtts/exp/YourTTS-Mandarin_Drama-September-04-2024_09+34AM-dbf1a08a/test_audio"
model_path="/mnt/md1/user_wago/TTS/recipes/mandarin_drama/yourtts/exp/YourTTS-Mandarin_Drama-September-04-2024_09+34AM-dbf1a08a/checkpoint_35000.pth"
config_path="/mnt/md1/user_wago/TTS/recipes/mandarin_drama/yourtts/exp/YourTTS-Mandarin_Drama-September-04-2024_09+34AM-dbf1a08a/config.json"
speakers_file_path="/mnt/md1/user_wago/TTS/recipes/mandarin_drama/yourtts/exp/YourTTS-Mandarin_Drama-September-04-2024_09+34AM-dbf1a08a/speakers.pth"

# 獲取 text_list 的長度
len=${#text_list[@]}

# 遍歷 text_list 和 speaker_idx_list
for (( i=0; i<$len; i++ )); do
    tts --text "${text_list[$i]}" \
        --out_path "${out_path}/${text_list[$i]}.wav" \
        --model_path "${model_path}" \
        --config_path "${config_path}" \
        --speakers_file_path "${speakers_file_path}" \
        --speaker_idx "${speaker_idx_list[$i]}"
done
