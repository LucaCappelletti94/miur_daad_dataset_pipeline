import json
from typing import Dict


def load_tasks(target:str)->Dict:
    """Return target project tasks.
        target: str, path from which to load the local tasks
    """
    tasks_path = "{path}/tasks.json".format(
        path=target
    )
    with open(tasks_path, "r") as f:
        return json.load(f)