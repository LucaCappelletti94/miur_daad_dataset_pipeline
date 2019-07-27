from .ungzip import ungzip
import os
from typing import List
from auto_tqdm import tqdm
from glob import glob
from multiprocessing import cpu_count, Pool

def ungzip_data(target: str):
    paths = list(glob('{target}/**/*.gz'.format(target=target), recursive=True))
    paths = [
            path for path in paths if not os.path.exists(path.split(".gz")[0])
    ]
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(ungzip, paths), desc="Expanding data", total=len(paths)))