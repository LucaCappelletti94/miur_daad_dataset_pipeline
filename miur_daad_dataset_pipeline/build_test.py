import pandas as pd
import numpy as np
from distutils.dir_util import copy_tree
from miur_daad_dataset_pipeline.utils import ungzip_data, get_cell_lines
from auto_tqdm import tqdm
import os
from typing import List
from auto_tqdm import tqdm
from glob import glob
from multiprocessing import cpu_count, Pool
import gzip
import shutil

def reduce(percentage:float, cell_line:str, target:str):
    labels_path = "{target}/classes/{cell_line}.csv".format(
        target=target,
        cell_line=cell_line
    )
    epigenomic_data_path = "{target}/epigenomic_data/{cell_line}.csv".format(
        target=target,
        cell_line=cell_line
    )
    regions_path = "{target}/regions/{cell_line}.bed".format(
        target=target,
        cell_line=cell_line
    )
    labels = pd.read_csv(labels_path, header=None)
    epigenomic_data = pd.read_csv(epigenomic_data_path, index_col=0)
    regions = pd.read_csv(regions_path, header=None, sep="\t")
    mask = np.zeros(labels.shape)
    mask[:int(mask.size*percentage)] = 1
    mask = mask.astype(bool)
    np.random.shuffle(mask)
    labels[mask].to_csv(labels_path, index=None, header=None)
    epigenomic_data[mask].to_csv(epigenomic_data_path)
    regions[mask].to_csv(regions_path, sep="\t", index=None, header=None)

def build_test(source_dataset:str, test_dataset:str, percentage:float = 0.001):
    copy_tree(source_dataset, test_dataset)
    ungzip_data(test_dataset)

    for cell_line in tqdm(get_cell_lines(test_dataset), desc="Reducing dataset to given percentage"):
        reduce(percentage, cell_line, test_dataset)