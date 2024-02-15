import pandas as pd
import os

from typing import Optional
from string import ascii_uppercase

SAVE_RESULTS_SCHEMA = {"image_path": [], "gpt4-v response": []}

BREAK_WORDS: list[str] = [
    "answer is (",
    "answer is  (",
    "answer is: (",
    "answer is:(",
    "answer is:  (",
    "answer is:\n(",
    "answer is: \n(",
    "answer is:\n\n(",
    "answer is: ",
    "answer is ",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "choices is: " r"is: $\boxed{\textbf{(",
    "answer: ",
    "answer is ",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is: $\boxed{\text{(",
]

def check_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
def check_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
def save_csv(df, path):
    df.to_csv(path, index=False)

def load_csv(path):
    return pd.read_csv(path)

def load_df(path, filename):
    if not os.path.exists(path + filename):
        return pd.DataFrame(SAVE_RESULTS_SCHEMA)
    else:
        return load_csv(path + filename)

def extract_answer(model_answer: str) -> Optional[str]:
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """
    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        # Sometimes there is a space in front of the answer
        last_item = tmp[-1].lstrip()

        if not last_item:
            continue

        ans = last_item[0]
        if ans in ascii_uppercase:
            return ans
    return None