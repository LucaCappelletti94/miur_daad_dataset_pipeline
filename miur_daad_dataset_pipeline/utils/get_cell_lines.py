from typing import List
import os

def get_cell_lines(target:str)->List[str]:
    """Return list of available cell_lines in given target."""
    return [
        f.split(".bed.gz")[0] for f in os.listdir(
            "{target}/regions".format(target=target)
        ) if f.endswith(".bed.gz")
    ]