from .utils import expand_regions, one_hot_encode_expanded_regions, ungzip_data, get_cell_lines
from ucsc_genomes_downloader import download_genome
import os

def build(target:str=None):
    target = "{script_directory}/dataset".format(
        script_directory=os.path.dirname(os.path.abspath(__file__))
    ) if target is None else target
    download_genome("hg19", path=target)
    ungzip_data(target)
    cell_lines = get_cell_lines(target)
    expand_regions(target, "hg19", cell_lines)
    one_hot_encode_expanded_regions(target, cell_lines)