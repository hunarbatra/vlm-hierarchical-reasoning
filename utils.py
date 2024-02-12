import pandas as pd

import os

SAVE_RESULTS_SCHEMA = {"image_path": [], "gpt4-v response": []}

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