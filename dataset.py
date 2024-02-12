import os
import random
import shutil

from typing import List

def get_dataset(dataset_name: str = 'imagenet-1k-1000', dataset_cap: int = 100) -> list[str]:
    dir_path = f"./datasets/{dataset_name}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    sampled_files = random.sample(all_files, dataset_cap)
    return sampled_files

def get_ref_dir_dataset(dataset_name: str = '', ref_dir: str = '') -> list[str]:
    dir_path = f"./experiments/{ref_dir}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    all_files = [file for file in all_files if '.jpg' not in file]
    all_files = [file for file in all_files if '.csv' not in file]
    all_files = [file for file in all_files if 'sam' not in file]
    return all_files

def check_file_exists(exp_dir: str = '') -> list:
    dir_path = f"./experiments/{exp_dir}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    all_files = [file for file in all_files if '.csv' not in file]
    all_files = [file.split('/')[-1] for file in all_files]
    all_files = [file for file in all_files if '-' not in file]
    all_files = [file for file in all_files if 'sam' not in file]
    return all_files

def format_question(question: str, options: List[str]) -> str:
    question = f'Q. {question}\n\nAnswer choices:\n'
    for i, option in enumerate(options):
        # question + A-Z in caps + . + option + \n
        question += f'{chr(65 + i)}. {option}\n'
    return question    
    