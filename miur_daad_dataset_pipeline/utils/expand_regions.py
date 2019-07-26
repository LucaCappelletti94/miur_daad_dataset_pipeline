import subprocess
import os
from auto_tqdm import tqdm
from typing import Dict

def expand_regions(target: str, genome: str, settings:Dict):
    """Expand the genomic regions using data withing given target and genome."""
    os.makedirs(
        "{target}/expanded_regions".format(target=target), exist_ok=True)
    for region in tqdm(settings["cell_lines"], desc="Expanding cell lines sequences"):
        goal = "{target}/expanded_regions/{region}.fa".format(
            region=region,
            target=target
        )
        region_path = "{target}/regions/{region}.bed".format(
            region=region,
            target=target
        )
        if not os.path.exists(goal):
            subprocess.run(
                ["fastaFromBed", "-fi", "{target}/{genome}.fa".format(genome=genome, target=target), "-bed", region_path, "-fo", goal])
