import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from auto_tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from notipy_me import Notipy
from MulticoreTSNE import MulticoreTSNE as TSNE
from .load import tasks_generator, balanced_holdouts_generator
from multiprocessing import cpu_count
from humanize import naturaldate
import os
import time
import numpy as np
from typing import Dict


def plot_clusters(df: pd.DataFrame, classes: pd.DataFrame, axis, title: str, std):
    colors = ["orange", "blue"]
    mask = (df.abs() < 2*std).any(axis=1)
    df, classes = df[mask], classes[mask]
    clustered = pd.concat([df, classes], axis=1)
    for i, label in enumerate(set(clustered.labels.values)):
        clustered[clustered.labels == label].plot(
            kind="scatter",
            x=clustered.columns[0],
            y=clustered.columns[1],
            color=colors[i],
            label=label,
            ax=axis,
            alpha=0.4
        )
    axis.set_title(title)


def clusterize(method, train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame, train_axes, test_axes, title: str):
    one, two = "First component", 'Second component'
    method.fit(train_x)
    scaler = MinMaxScaler()
    train = pd.DataFrame(data=method.transform(train_x), columns=[one, two])
    test = pd.DataFrame(data=method.transform(test_x), columns=[one, two])
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    std = train_x.std()
    plot_clusters(train_x, train_y, train_axes,
                  title.format(set_name="Training set"), std)
    plot_clusters(test_x, test_y, test_axes,
                  title.format(set_name="Testing set"), std)


def pca(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame, train_axes, test_axes):
    clusterize(
        PCA(n_components=2),
        train_x,
        train_y,
        test_x,
        test_y,
        train_axes,
        test_axes,
        "{set_name} PCA decomposition"
    )


def tsne(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame, train_axes, test_axes):
    clusterize(
        TSNE(n_jobs=cpu_count(), verbose=2),
        train_x,
        train_y,
        test_x,
        test_y,
        train_axes,
        test_axes,
        "{set_name} TSNE decomposition"
    )


def labelize(classes: np.ndarray, task: Dict) -> pd.DataFrame:
    labelized = pd.DataFrame(classes, columns=["labels"])
    labelized[labelized.labels == 1] = ", ".join(task["positive"])
    labelized[labelized.labels == 0] = ", ".join(task["negative"])
    return labelized


def visualize(target: str):
    with Notipy() as r:
        tasks = list(enumerate(tasks_generator(target)))
        for i, (target, cell_line, task, balance_mode) in tqdm(tasks):
            path = f"visualize/clustering/{cell_line}"
            if os.path.exists(path):
                continue
            os.makedirs(path, exist_ok=True)
            generator = balanced_holdouts_generator(target, cell_line, task, balance_mode, {
                "quantities": [1],
                "test_sizes": [0.3]
            }, verbose=False)
            ((train_x, train_y), (test_x, test_y)), _, _ = next(generator())
            train_y = labelize(train_y, task)
            test_y = labelize(test_y, task)
            _, axes = plt.subplots(1, 4, figsize=(8*4, 8))
            pca(train_x, train_y, test_x, test_y, axes[0], axes[1])
            tsne(train_x, train_y, test_x, test_y, axes[2], axes[3])
            title = "{cell_line}-{task}-{balance_mode}".format(
                task=task["name"],
                cell_line=cell_line,
                balance_mode=balance_mode
            )
            plt.tight_layout()
            plt.savefig(f"{path}/{title}.png".replace(" ", "_"))
            plt.close()
            r.add_report(pd.DataFrame({
                "cell line": cell_line,
                "task": task["name"],
                "balance_mode": balance_mode
            }, index=[i]))
