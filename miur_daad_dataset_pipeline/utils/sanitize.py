import numpy as np
import pandas as pd
from typing import Tuple, Dict
from .load_csv import load_raw_classes, load_raw_epigenomic_data, load_raw_nucleotides_sequences
from .store_csv import store_raw_classes, store_raw_epigenomic_data, store_raw_nucleotides_sequences
from auto_tqdm import tqdm

def drop_unknown_datapoints(epigenomic_data:pd.DataFrame, nucleotides_sequences:np.ndarray, nucleotides_sequences_index:np.ndarray, classes:pd.DataFrame)->Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Remove datapoints labeled as unknown (UK)."""
    unknown = classes["UK"] == 1
    epigenomic_data = epigenomic_data.drop(index=epigenomic_data.index[unknown])
    nucleotides_sequences = nucleotides_sequences[~unknown]
    nucleotides_sequences_index = nucleotides_sequences_index[~unknown]
    classes = classes.drop(index=classes.index[unknown])
    classes = classes.drop(columns=["UK"])
    return epigenomic_data, nucleotides_sequences, nucleotides_sequences_index, classes

def sanitize(target:str, settings:Dict):
    for cell_line in tqdm(settings["cell_lines"], desc="Sanitizing data"):
        classes = load_raw_classes(target, cell_line)
        if "UK" not in classes.columns:
            continue
        epigenomic_data = load_raw_epigenomic_data(target, cell_line)
        nucleotides_sequences, nucleotides_sequences_index, nucleotides_sequences_columns = load_raw_nucleotides_sequences(target, cell_line)
        epigenomic_data, nucleotides_sequences, nucleotides_sequences_index, classes = drop_unknown_datapoints(epigenomic_data, nucleotides_sequences, nucleotides_sequences_index, classes)
        store_raw_epigenomic_data(target, cell_line, epigenomic_data)
        store_raw_nucleotides_sequences(target, cell_line, nucleotides_sequences, nucleotides_sequences_index, nucleotides_sequences_columns)
        store_raw_classes(target, cell_line, classes)