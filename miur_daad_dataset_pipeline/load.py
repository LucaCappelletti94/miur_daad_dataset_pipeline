import pandas as pd
from typing import Dict, Callable, Generator
from .utils import load_settings, balance, load_raw_nucleotides_sequences, load_raw_classes, load_raw_epigenomic_data, load_hostnames
from holdouts_generator import random_holdouts, holdouts_generator
import socket


def balanced_generator(generator, mode: str, pos: str, neg: str, settings: Dict)->Callable:
    if generator is None:
        return None

    def wrapper():
        for (training, testing), sub_generator in generator():
            training, testing = balance(
                training, testing, mode, pos, neg, settings)
            yield (training, testing), balanced_generator(sub_generator, mode, pos, neg, settings)
    return wrapper


def balanced_holdouts_generator(target: str, cell_line: str, task: Dict, balance_mode: str):
    settings = load_settings(target)
    epigenomic_data = load_raw_epigenomic_data(target, cell_line)
    nucleotides_sequences, _, _ = load_raw_nucleotides_sequences(
        target, cell_line)
    classes = pd.DataFrame(load_raw_classes(target, cell_line)[
                           task["positive"]].any(axis=1), columns=["+".join(task["positive"])])
    generator = holdouts_generator(
        epigenomic_data, nucleotides_sequences, classes,
        holdouts=random_holdouts(**settings["holdouts"]),
        cache=True,
        cache_dir=".holdouts/{target}/{cell_line}/{name}".format(
            target=target,
            cell_line=cell_line,
            name=task["name"].replace(" ", "_")
        )
    )
    return balanced_generator(generator, balance_mode, "+".join(task["positive"]), "+".join(task["negative"]), settings["balance"])


def tasks_generator(target: str)->Generator:
    settings = load_settings(target)
    hostnames = load_hostnames(target)
    hostname = socket.gethostname()
    return (
        (target, cell_line, task, balance_mode)
        for cell_line in settings["cell_lines"]
        if hostname not in hostnames or cell_line in hostnames[hostname]["cell_lines"]
        for task in settings["tasks"]
        if task["enabled"] and (hostname not in hostnames or task["name"] in hostnames[hostname]["tasks"])
        for balance_mode, mode_enabled in task["balancing"].items()
        if mode_enabled and (hostname not in hostnames or balance_mode in hostnames[hostname]["balancing"])
    )
