from .utils import expand_regions, one_hot_encode_expanded_regions, ungzip_data, load_cell_lines
from ucsc_genomes_downloader import download_genome
import os

def build(target:str):
    download_genome("hg19", path=target)
    ungzip_data(target)
    cell_lines = load_cell_lines(target)
    expand_regions(target, "hg19", cell_lines)
    one_hot_encode_expanded_regions(target, cell_lines)