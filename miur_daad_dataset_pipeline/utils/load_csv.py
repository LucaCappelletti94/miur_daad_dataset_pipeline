import pandas as pd
from typing import Tuple
from .paths import get_raw_epigenomic_data_path, get_raw_nucleotides_sequences_path, get_raw_classes_path

def load_csv(path:str):
    return pd.read_csv(path, index_col=0)

load_epigenomic_data = load_classes = load_csv

def load_nucleotides_sequences(path:str)->Tuple:
    return load_csv(path).values.reshape(-1, 200, 5)

def load_raw_epigenomic_data(target:str, cell_line:str):
    return load_epigenomic_data(get_raw_epigenomic_data_path(target, cell_line))

def load_raw_nucleotides_sequences(target:str, cell_line:str):
    return load_nucleotides_sequences(get_raw_nucleotides_sequences_path(target, cell_line))

def load_raw_classes(target:str, cell_line:str):
    return pd.read_csv(get_raw_classes_path(target, cell_line), header=None)