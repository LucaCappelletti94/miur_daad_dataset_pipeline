from auto_tqdm import tqdm
from fasta_one_hot_encoder import FastaOneHotEncoder
import os
from typing import List
from .ungzip import ungzip


def one_hot_encode_expanded_regions(target: str, cell_lines:List[str]):
    os.makedirs(
        "{target}/one_hot_encoded_expanded_regions".format(target=target), exist_ok=True)
    encoder = FastaOneHotEncoder(
        nucleotides="acgtn",
        kmers_length=1,
        lower=True,
        sparse=False
    )
    for cell_line in tqdm(cell_lines, leave=False, desc="One-hot encode nucleotides"):
        path = "{target}/one_hot_encoded_expanded_regions/{cell_line}.csv".format(
            cell_line=cell_line,
            target=target
        )
        if os.path.exists(path):
            continue
        expand_cell_line_path = "{target}/expanded_regions/{cell_line}.fa".format(
            cell_line=cell_line,
            target=target
        )
        encoder.transform_to_df(expand_cell_line_path).to_csv(path)