from json import load
from typing import Dict, List


def load_json(target: str, file_path: str)->Dict:
    """Return given target settings for given json file_path.
        target: str, path for the current target.
        file_path:str, name of the file.
    """
    with open("{target}/{file_path}.json".format(
        target=target,
        file_path=file_path
    ), "r") as f:
        return load(f)


def load_holdouts(target: str)->Dict:
    """Return target project holdouts.
        target: str, path from which to load the local holdouts.
    """
    return load_json(target, "holdouts")


def load_gaussian_process_holdouts(target: str)->Dict:
    """Return target project gaussian process holdouts.
        target: str, path from which to load the local holdouts.
    """
    return load_json(target, "gaussian_process_holdouts")


def load_tasks(target: str)->Dict:
    """Return target project tasks.
        target: str, path from which to load the local tasks.
    """
    return load_json(target, "tasks")


def load_cell_lines(target: str)->List[str]:
    """Return list of enabled cell_lines in given target.
        target: str, path from which to load the local tasks.
    """
    return [
        k for k, v in load_json(target, "cell_lines").items() if v
    ]
