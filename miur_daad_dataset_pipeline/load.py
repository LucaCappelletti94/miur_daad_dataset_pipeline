import pandas as pd
from typing import Dict, Callable, Generator, List
from .utils import load_holdouts, get_cell_lines, load_tasks, load_raw_nucleotides_sequences, load_raw_classes, load_raw_epigenomic_data
from holdouts_generator import random_holdouts, cached_holdouts_generator, skip
from miur_daad_balancing import get_callback
import numpy as np
import os

def balanced_generator(generator:Generator, balance: Callable, positive:List[str]) -> Callable:
    if generator is None:
        return None

    def wrapper(*args, **kwargs):
        for (training, testing), key, sub_generator in generator(*args, **kwargs):
            if training is not None:
                training, testing = balance(training, testing)
                training = *training[:2], np.array([
                    c in positive for c in training[-1]
                ])
                testing = *testing[:2], np.array([
                    c in positive for c in testing[-1]
                ])
            yield (training, testing), key, balanced_generator(sub_generator, balance, positive)
    return wrapper


def balanced_holdouts_generator(target: str, cell_line: str, task: Dict, balance_mode: str):
    epigenomic_data = load_raw_epigenomic_data(target, cell_line).values
    nucleotides_sequences = load_raw_nucleotides_sequences(target, cell_line)
    classes = load_raw_classes(target, cell_line).values
    used_classes = task["positive"] + task["negative"]
    mask = np.array([
        c in used_classes for c in classes
    ])
    epigenomic_data = epigenomic_data[mask]
    nucleotides_sequences = nucleotides_sequences[mask]
    classes = classes[mask]

    generator = cached_holdouts_generator(
        epigenomic_data,
        nucleotides_sequences,
        classes,
        holdouts=random_holdouts(**load_holdouts(target)),
        skip=skip,
        cache_dir=".holdouts/{target}/{cell_line}/{name}".format(
            target=target,
            cell_line=cell_line,
            name=task["name"].replace(" ", "_")
        )
    )
    return balanced_generator(generator, get_callback(balance_mode), task["positive"])


def tasks_generator(target: str=None) -> Generator:
    if target is None:
        target = "{script_directory}/dataset".format(
            script_directory=os.path.dirname(os.path.abspath(__file__))
        )
    tasks = load_tasks(target)
    return (
        (target, cell_line, task, balance_mode)
        for cell_line in get_cell_lines(target)
        for task in tasks
        for balance_mode, mode_enabled in task["balancing"].items()
        if mode_enabled
    )
