import os
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
import pandas as pd

# 計算餘弦相似度的函數
def compute_cosine_similarity(emb1, emb2):
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()
    return F.cosine_similarity(emb1, emb2, dim=0).item()

# 初始化
spk_list = []
rep_emb_map = {}
sum_emb_map = {}
count_map = {}  # 新增一個 map 記錄每個語者的句子數量
threshold = 0.3  # 相似度閥值，可根據需求調整

def count_first_level_subdirs(directory):
    for root, dirs, files in os.walk(directory):
        return len(dirs)  # 只返回第一層子目錄的數量

def process_embeddings(directory, output_csv, csv_path):
    df = pd.read_csv(csv_path)
    # filename_list = df['filename'].values.split('/')[-1].split('.')[0]
    # filename_list = df['filename'].apply(lambda x: x.split('/')[-1].split('.')[0][12:]).tolist()
    # print(f'filename_list: {filename_list}')
    # return
    results = []
    file_index = 1
    folder_index = 0
    folder_num = count_first_level_subdirs(directory)
    for root, _, files in os.walk(directory):
        print(f'processing {folder_index}/{folder_num} folder..., spekaer len: {len(spk_list)}')
        for file in tqdm(files):
            # print(f'file: {file}')
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                embedding = torch.load(file_path).squeeze()
                
                max_similarity = -1
                best_spk = None

                # 比較與每個現有語者的相似度
                for spk in spk_list:
                    rep_emb = rep_emb_map[spk]
                    similarity = compute_cosine_similarity(rep_emb, embedding)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_spk = spk

                # 如果相似度超過閥值，則更新該語者
                if max_similarity >= threshold:
                    sum_emb_map[best_spk] += embedding
                    count_map[best_spk] += 1
                    rep_emb_map[best_spk] = sum_emb_map[best_spk] / count_map[best_spk]
                    results.append([file, best_spk, max_similarity])
                    # print(f"File {file} classified as speaker {best_spk} with similarity {max_similarity}")
                else:
                    # 創建新語者
                    new_spk = len(spk_list) + 1
                    spk_list.append(new_spk)
                    sum_emb_map[new_spk] = embedding.clone()
                    rep_emb_map[new_spk] = embedding.clone()
                    count_map[new_spk] = 1  # 初始化語者的句子數量為 1
                    results.append([file, new_spk, max_similarity])
                    # print(f"File {file} classified as new speaker {new_spk}")
            file_index += 1
            if file_index % 20000 == 0:
                with open(output_csv, mode='w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['File', 'Speaker', 'Max Similarity'])  # 寫入表頭
                    csv_writer.writerows(results)
            # print(f'file_index: {file_index}')
        folder_index += 1
        
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File', 'Speaker', 'Max Similarity'])  # 寫入表頭
        csv_writer.writerows(results)

# 指定要讀取的目錄和輸出的 CSV 檔案路徑
directory = '/mnt/md1/user_wago/data/mandarin_drama/speaker_emb'
output_csv = f'speaker_label_threshold_{threshold}_95hr_select.csv'
csv_path = '/mnt/md1/user_wago/MOS/csv/mandarin_drama_95hr_sub_select.csv'

# 開始處理嵌入
process_embeddings(directory, output_csv, csv_path)

# 輸出語者列表
print(f"Total number of speakers: {len(spk_list)}")
