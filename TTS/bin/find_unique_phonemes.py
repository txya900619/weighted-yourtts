"""Find all the unique characters in a dataset"""
import argparse
import multiprocessing
from argparse import RawTextHelpFormatter

from tqdm.contrib.concurrent import process_map

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.phonemizers import Gruut


def compute_phonemes(item):
    text = item["text"]
    ph = phonemizer.phonemize(text).replace("|", "")
    return set(list(ph))


def main():
    # pylint: disable=W0601
    global c, phonemizer
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_phonemes.py --config_path config.json
    """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, help="Path to dataset config file.", required=True)
    args = parser.parse_args()

    c = load_config(args.config_path)

    # load all datasets
    train_items, eval_items = load_tts_samples(
        c.datasets, eval_split=True, eval_split_max_size=c.eval_split_max_size, eval_split_size=c.eval_split_size
    )
    items = train_items + eval_items
    print("Num items:", len(items))

    language_list = [item["language"] for item in items]
    is_lang_def = all(language_list)

    if not c.phoneme_language or not is_lang_def:
        raise ValueError("Phoneme language must be defined in config.")

    if not language_list.count(language_list[0]) == len(language_list):
        raise ValueError(
            "Currently, just one phoneme language per config file is supported !! Please split the dataset config into different configs and run it individually for each language !!"
        )

    phonemizer = Gruut(language=language_list[0], keep_puncs=True)

    phonemes = process_map(compute_phonemes, items, max_workers=multiprocessing.cpu_count(), chunksize=15)
    phones = []
    for ph in phonemes:
        phones.extend(ph)

    phones = set(phones)
    lower_phones = filter(lambda c: c.islower(), phones)
    phones_force_lower = [c.lower() for c in phones]
    phones_force_lower = set(phones_force_lower)

    print(f" > Number of unique phonemes: {len(phones)}")
    print(f" > Unique phonemes: {''.join(sorted(phones))}")
    print(f" > Unique lower phonemes: {''.join(sorted(lower_phones))}")
    print(f" > Unique all forced to lower phonemes: {''.join(sorted(phones_force_lower))}")


    contents = [
        f" > Number of unique characters: {len(phones)}\n",
        f" > Unique characters: {''.join(sorted(phones))}\n",
        f" > Unique lower characters: {''.join(sorted(lower_phones))}\n",
        f" > Unique all forced to lower characters: {''.join(sorted(phones_force_lower))}\n"
    ]
    
    # 定義文件名稱
    file_names = [
        'unique_phones_count.txt',
        'unique_phones.txt',
        'unique_lower_phones.txt',
        'unique_phones_force_lower.txt'
    ]

    # 寫入內容到各自的文件
    for content, file_name in zip(contents, file_names):
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)

    print("All files have been written successfully.")

if __name__ == "__main__":
    main()
