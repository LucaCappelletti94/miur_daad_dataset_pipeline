import json
from typing import Dict

def load_holdouts(target:str)->Dict:
    """Return target project holdouts.
        target: str, path from which to load the local holdouts
    """
    holdouts_path = "{path}/holdouts.json".format(
        path=target
    )
    with open(holdouts_path, "r") as f:
        return json.load(f)