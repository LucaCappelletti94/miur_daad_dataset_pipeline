import subprocess
import os
from auto_tqdm import tqdm
from typing import List


def expand_regions(target: str, genome: str, cell_lines: List[str]):
    """Expand the genomic regions using data withing given target and genome."""
    os.makedirs(
        "{target}/expanded_regions".format(target=target), exist_ok=True)
    for cell_line in tqdm(cell_lines, desc="Expanding cell lines sequences"):
        goal = "{target}/expanded_regions/{cell_line}.fa".format(
            cell_line=cell_line,
            target=target
        )
        region_path = "{target}/regions/{cell_line}.bed".format(
            cell_line=cell_line,
            target=target
        )
        if not os.path.exists(goal):
            subprocess.run(
                ["fastaFromBed", "-fi", "{target}/{genome}.fa".format(genome=genome, target=target), "-bed", region_path, "-fo", goal])
