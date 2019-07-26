import pandas as pd
import os
from auto_tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from typing import Dict
from .ungzip import ungzip


def one_hot_encode(classes, filename):
    encoder = OneHotEncoder(categories='auto', sparse=False)
    encoder.fit(classes.reshape(-1, 1))
    one_hot_encoded = encoder.transform(classes.reshape(-1, 1))
    return pd.DataFrame(
        one_hot_encoded, columns=encoder.categories_,
        dtype="int").to_csv(filename)


def one_hot_encode_classes(target: str, settings:Dict):
    os.makedirs(
        "{target}/one_hot_encoded_classes".format(target=target), exist_ok=True)
    for region in tqdm(settings["cell_lines"], desc="One-hot encode classes"):
        region_classes = "{target}/classes/{region}.csv".format(
            region=region,
            target=target
        )
        path = "{target}/one_hot_encoded_classes/{region}.csv".format(
            region=region,
            target=target
        )
        if os.path.exists("{path}.gz".format(path=path)):
            ungzip("{path}.gz".format(path=path))
            continue
        if not os.path.exists(path):
            classes = pd.read_csv(region_classes, header=None)
            one_hot_encode(classes[0].ravel(), path)
