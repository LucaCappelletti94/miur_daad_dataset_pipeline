import shutil
import os
from auto_tqdm import tqdm
from glob import glob
from multiprocessing import cpu_count, Pool


def clear(target: str):
    """Clear all generated data for given target, holdouts cache included.
        target:str, path to dataset to clear.
    """
    dirs = (
        "one_hot_encoded_expanded_regions",
        "expanded_regions"
    )
    for d in dirs:
        shutil.rmtree(
            "{target}/{d}".format(target=target, d=d), ignore_errors=True
        )
    paths = list(glob('{target}/**/*.csv'.format(target=target), recursive=True))
    paths += list(glob('{target}/**/*.bed'.format(target=target), recursive=True))
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(os.remove, paths), desc="Compress data", total=len(paths)))
        p.close()
        p.join()