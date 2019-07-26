import pandas as pd
import numpy as np
from .paths import get_raw_epigenomic_data_path, get_raw_nucleotides_sequences_path, get_raw_classes_path


def store_csv(path: str, data: np.ndarray, index: np.ndarray, columns: np.ndarray):
    pd.DataFrame(
        data=data,
        index=index,
        columns=columns
    ).to_csv(path)


def store_nucleotides_sequences(path: str, data: np.ndarray, index: np.ndarray, columns: np.ndarray):
    store_csv(
        path,
        data.reshape(-1, 5),
        index.reshape(-1),
        columns
    )


def store_raw_epigenomic_data(target: str, cell_line: str, epigenomic_data: pd.DataFrame):
    return store_csv(get_raw_epigenomic_data_path(target, cell_line), epigenomic_data.values, epigenomic_data.index, epigenomic_data.columns)


def store_raw_nucleotides_sequences(target: str, cell_line: str, data: np.ndarray, index: np.ndarray, columns: np.ndarray):
    return store_nucleotides_sequences(get_raw_nucleotides_sequences_path(target, cell_line), data, index, columns)


def store_raw_classes(target: str, cell_line: str, classes: pd.DataFrame):
    return store_csv(get_raw_classes_path(target, cell_line), classes.values, classes.index, classes.columns)
