from typing import List
from json import load

def get_cell_lines(target:str)->List[str]:
    """Return list of enabled cell_lines in given target."""
    with open("{target}/cell_lines.json".format(target=target), "r") as f:
        return [
            k for k, v in load(f).items() if v
        ]