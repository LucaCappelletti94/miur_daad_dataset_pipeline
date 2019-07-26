from .ungzip import ungzip
import os
from typing import Dict
from auto_tqdm import tqdm

def ungzip_data(target: str, settings:Dict):
    for cell_line in tqdm(settings["cell_lines"], desc="Expanding epigenomic data"):
        path =  "{target}/epigenomic_data/{cell_line}.csv".format(
            target=target,
            cell_line=cell_line
        )
        if os.path.exists(path):
            continue
        ungzip("{path}.gz".format(path=path))