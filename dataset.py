import os
import random
import shutil

import ast

from typing import List

def get_dataset(dataset_name: str = 'imagenet-1k-1000', dataset_cap: int = 100, files_to_include: list[str] = [], dataset_cap_max: bool = False) -> list[str]:
    dir_path = f"./datasets/{dataset_name}"
    if not len(files_to_include):
        all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    else: 
        all_files = [os.path.join(dir_path, file) for file in files_to_include]
    if dataset_cap_max:
        dataset_cap = len(all_files)
    sampled_files = random.sample(all_files, dataset_cap)
    return sampled_files

def get_ref_dir_dataset(dataset_name: str = '', ref_dir: str = '', check_sam: bool = True) -> list[str]:
    dir_path = f"./experiments/{ref_dir}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    all_files = [file for file in all_files if '.jpg' not in file]
    all_files = [file for file in all_files if '.csv' not in file]
    if check_sam:
        all_files = [file for file in all_files if 'sam' not in file]
    return all_files

def check_file_exists(exp_dir: str = '', check_sam: bool = True) -> list:
    dir_path = f"./experiments/{exp_dir}"
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    all_files = [file for file in all_files if '.csv' not in file]
    all_files = [file.split('/')[-1] for file in all_files]
    all_files = [file for file in all_files if '-' not in file]
    if check_sam:
        all_files = [file for file in all_files if 'sam' not in file]
    return all_files

def format_question(question: str, options: List[str]) -> str:
    question = f'Q. {question}\n\nAnswer choices:\n'
    options = ast.literal_eval(options)
    for index, option in enumerate(options):
        mcq_letter = chr(65 + index)
        question += f'{mcq_letter}. {option}\n'
    return question
    