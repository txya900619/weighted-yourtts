from huggingface_hub import hf_hub_download
import torch
import torchaudio
import pandas as pd
import os
from tqdm import tqdm 

def inference_from_mos_result(model, mos_result_file, output_base, device="cpu"):
    df = pd.read_csv(mos_result_file)
    column_name = 'filename'  

    for value in tqdm(df[column_name]):
        basename = os.path.basename(value)
        sub_folder = os.path.join(output_base, basename[:3]) 
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        output  = os.path.join(sub_folder, basename.replace('wav','pt'))
        if os.path.exists(output):
            # print(f'{basename} already exist, skipped')
            continue
        audio, sr = torchaudio.load(value) # sample rate of 16 kHz expected
        if sr != 16000:
            raise("sample error, 16 kHz expected")
        embedding = model(audio.to(device))
        # print(f'embedding: {embedding}')
        # print(f'tupe(embedding): {type(embedding)}')
        # print(f'value: {value}')
        # print(f'basename: {basename}')
        # print(f'output: {output}')
        torch.save(embedding, output)
        # return 
        
def inference_to_mos(model, mos_result_file, output_base, output_csv, device="cpu"):
    df = pd.read_csv(mos_result_file)
    column_name = 'name'  
    
    # df['embedding'] = None
    # import re
    # df['pinyin'] = df['pinyin'].apply(lambda x: re.sub(r"[,\[\]\"']|None", " ", str(x)).strip())
    # df['pinyin'] = df['pinyin'].apply(lambda x: re.sub(r"\s+", " ", x))
    # df.to_csv(output_csv, index=False)
    model = model.to(device)
    for index, value in tqdm(df[column_name].items()):
        basename = os.path.basename(value)
        sub_folder = os.path.join(output_base, basename[:11]) 
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        output = os.path.join(sub_folder, basename.replace('wav', 'pt'))
        if os.path.exists(output):
            # print(f'{basename} already exist, skipped')
            continue
        audio, sr = torchaudio.load(value)  # 預期的採樣率為16 kHz
        if sr != 16000:
            raise ValueError("sample error, 16 kHz expected")
        
        embedding = model(audio.to(device)).cpu().detach()
        
        # 保存嵌入到指定位置
        torch.save(embedding, output)
        
        # 同時將嵌入結果存入 DataFrame 的 'embedding' 欄位
        # df.at[index, 'embedding'] = embedding.numpy()
    

if __name__ == '__main__':
    # automatically checks for cached file, optionally set `cache_dir` location
    model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)

    ecapa2 = torch.jit.load(model_file, map_location='cuda')
    ecapa2.half() # optional, but results in faster inference
    
    # mos_result_file = '/mnt/md1/user_wago/MOS/csv/mandarin_drama_combined.csv.csv'
    output_base = '/mnt/md1/user_wago/data/mandarin_drama/combined_select_spk'
    # inference_from_mos_result(ecapa2,mos_result_file,output_base,"cuda")
    output_csv = 'mandarin_drama_combined_select_spk_pinyin.csv'
    mos_result_file = '/mnt/md1/user_wago/data/mandarin_drama/combined_mandarin_drama_pinyin.csv'
    inference_to_mos(ecapa2,mos_result_file,output_base,output_csv,"cuda")