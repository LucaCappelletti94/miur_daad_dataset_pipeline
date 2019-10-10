import pandas as pd
from typing import Dict, Callable, Generator, List, Tuple
from .utils import load_cell_lines, load_tasks, load_raw_nucleotides_sequences, load_raw_classes, load_raw_epigenomic_data
from holdouts_generator import random_holdouts, cached_holdouts_generator, skip
from miur_daad_balancing import get_callback
from multiprocessing import cpu_count, Pool
from notipy_me import Notipy
from auto_tqdm import tqdm
import numpy as np
import os

def balanced_generator(generator:Generator, balance: Callable, positive:List[str]) -> Callable:
    if generator is None:
        return None

    def wrapper(*args, **kwargs):
        for (training, testing), key, sub_generator in generator(*args, **kwargs):
            if training is not None:
                training, testing = balance(training, testing)
                training = *training[:-1], np.array([
                    c in positive for c in training[-1]
                ])
                testing = *testing[:-1], np.array([
                    c in positive for c in testing[-1]
                ])
            yield (training, testing), key, balanced_generator(sub_generator, balance, positive)
    return wrapper


def balanced_holdouts_generator(target: str, cell_line: str, task: Dict, balance_mode: str, holdouts:Dict, verbose:bool=True, cache_dir:str=".holdouts"):
    classes = load_raw_classes(target, cell_line).values
    used_classes = task["positive"] + task["negative"]
    mask = np.array([
        c in used_classes for c in classes
    ])

    data = []

    if task["epigenomic_data"]:
        epigenomic_data = load_raw_epigenomic_data(target, cell_line).values
        epigenomic_data = epigenomic_data[mask]
        data.append(epigenomic_data)

    if task["nucleotides_sequences"]:
        nucleotides_sequences = load_raw_nucleotides_sequences(target, cell_line)
        nucleotides_sequences = nucleotides_sequences[mask]
        data.append(nucleotides_sequences)
    
    classes = classes[mask]

    generator = cached_holdouts_generator(
        *data,
        classes,
        holdouts=random_holdouts(**holdouts, hyper_parameters=task),
        skip=skip,
        cache_dir=cache_dir,
        verbose=verbose
    )
    return balanced_generator(generator, get_callback(balance_mode), task["positive"])

def task_builder(target:str, holdouts:Dict, cache_dir:str=".holdouts"):
    with Notipy() as report:
        tasks = list(tasks_generator(target))
        for i, task in tqdm(enumerate(tasks), total=len(tasks), desc="Build tasks"):
            generator = balanced_holdouts_generator(*task, holdouts, cache_dir=cache_dir)
            for _, _, sub_generator in generator():
                if sub_generator is not None:
                    for _ in sub_generator():
                        pass
            report.add_report(pd.DataFrame({
                "task":task[2]["name"],
                "cell_line":task[1],
                "balancing":task[3]
            }, index=[i]))

def tasks_generator(target: str) -> Generator:
    tasks = load_tasks(target)
    return (
        (target, cell_line, task, balance_mode)
        for cell_line in load_cell_lines(target)
        for task in tasks
        for balance_mode, mode_enabled in task["balancing"].items()
        if mode_enabled
    )
