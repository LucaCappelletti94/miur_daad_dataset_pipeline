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
    colors = ["#ff7f0e", "#1f77b4"]
    mask = (df < 6*std).any(axis=1)
    df, classes = df[mask], classes[mask]
    clustered = pd.concat([df, classes], axis=1)
    for i, label in enumerate(set(clustered.labels.values)):
        clustered[clustered.labels == label].plot(
            kind="scatter",
            edgecolors='none',
            x=clustered.columns[0],
            y=clustered.columns[1],
            color=colors[i],
            label=label,
            ax=axis,
            alpha=0.5
        )

    axis.set_title(title)


def clusterize(method, X: pd.DataFrame, y: pd.DataFrame, mask:np.array, train_axes, test_axes, title: str):
    one, two = "First component", 'Second component'
    scaler = MinMaxScaler()
    X = pd.DataFrame(data=scaler.fit_transform(method.fit_transform(X)), columns=[one, two])
    std = X.std()
    plot_clusters(X[mask], y[mask], train_axes,
                  title.format(set_name="Training set"), std)
    plot_clusters(X[~mask], y[~mask], test_axes,
                  title.format(set_name="Testing set"), std)


def pca(X: pd.DataFrame, y: pd.DataFrame, mask:np.array, train_axes, test_axes):
    clusterize(
        PCA(n_components=2, random_state=42),
        X,
        y,
        mask,
        train_axes,
        test_axes,
        "{set_name} PCA decomposition"
    )


def tsne(X: pd.DataFrame, y: pd.DataFrame, mask:np.array, train_axes, test_axes):
    clusterize(
        TSNE(n_jobs=cpu_count(), verbose=0, random_state=42),
        X,
        y,
        mask,
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
            title = "{cell_line}-{balance_mode}-{task}".format(
                task=task["name"],
                cell_line=cell_line,
                balance_mode=balance_mode.replace("umbalanced", "unbalanced")
            )
            if os.path.exists(f"{i}.tmp") or os.path.exists(f"{path}/{title}.png".replace(" ", "_")):
                continue
            with open(f"{i}.tmp", "w") as f:
                f.write("")
            os.makedirs(path, exist_ok=True)
            generator = balanced_holdouts_generator(target, cell_line, task, balance_mode, {
                "quantities": [1],
                "test_sizes": [0.3]
            }, verbose=False)
            ((train_x, train_y), (test_x, test_y)), _, _ = next(generator())
            X = np.vstack([train_x, test_x])
            y = labelize(np.hstack([train_y, test_y]), task)
            mask = np.zeros(y.size)
            mask[:train_y.size] = 1
            mask = mask.astype(bool)
            _, axes = plt.subplots(1, 4, figsize=(8*4, 8))
            pca(X, y, mask, axes[0], axes[1])
            tsne(X, y, mask, axes[2], axes[3]) 
            plt.tight_layout()
            plt.savefig(f"{path}/{title}.png".replace(" ", "_"))
            plt.close()
            os.remove(f"{i}.tmp")
            r.add_report(pd.DataFrame({
                "cell line": cell_line,
                "task": task["name"],
                "balance_mode": balance_mode
            }, index=[i]))
