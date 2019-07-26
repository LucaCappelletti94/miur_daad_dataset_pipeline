import json
from typing import Dict


def load_settings(target:str)->Dict:
    """Return target project settings.
        target: str, path from which to load the local settings
    """
    settings_path = "{path}/settings.json".format(
        path=target
    )
    with open(settings_path, "r") as f:
        return json.load(f)