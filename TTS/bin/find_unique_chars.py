"""Find all the unique characters in a dataset"""
import argparse
from argparse import RawTextHelpFormatter

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_chars.py --config_path config.json
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

    texts = "".join(item["text"] for item in items)
    chars = set(texts)
    lower_chars = filter(lambda c: c.islower(), chars)
    chars_force_lower = [c.lower() for c in chars]
    chars_force_lower = set(chars_force_lower)

    print(f" > Number of unique characters: {len(chars)}")
    print(f" > Unique characters: {''.join(sorted(chars))}")
    print(f" > Unique lower characters: {''.join(sorted(lower_chars))}")
    print(f" > Unique all forced to lower characters: {''.join(sorted(chars_force_lower))}")
    
    contents = [
        f" > Number of unique characters: {len(chars)}\n",
        f" > Unique characters: {''.join(sorted(chars))}\n",
        f" > Unique lower characters: {''.join(sorted(lower_chars))}\n",
        f" > Unique all forced to lower characters: {''.join(sorted(chars_force_lower))}\n"
    ]

    # 定義文件名稱
    file_names = [
        'unique_characters_count.txt',
        'unique_characters.txt',
        'unique_lower_characters.txt',
        'unique_characters_force_lower.txt'
    ]

    # 寫入內容到各自的文件
    for content, file_name in zip(contents, file_names):
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)

    print("All files have been written successfully.")


if __name__ == "__main__":
    main()
