import json
import os
from typing import Dict


def load_hostnames(target:str)->Dict:
    """Return target project hostnames.
        target: str, path from which to load the local hostnames
    """
    hostnames_path = "{path}/hostnames.json".format(
        path=target
    )
    if os.path.exists(hostnames_path):
        with open(hostnames_path, "r") as f:
            return json.load(f)
    return {}