from typing import Dict
from mca import MCA
import numpy as np
import time
import os
from humanize import naturaldate
from multiprocessing import cpu_count
from .load import tasks_generator, balanced_holdouts_generator
from MulticoreTSNE import MulticoreTSNE as TSNE
from notipy_me import Notipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pylab
from matplotlib import pyplot as plt
from auto_tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
matplotlib.use('Agg')


def plot_clusters(df: pd.DataFrame, classes: pd.DataFrame, axis, title: str):
    colors = ["#ff7f0e", "#1f77b4"]
    for i, label in enumerate(set(classes.values.flatten())):
        label_mask = classes.values.flatten() == label
        df[label_mask].plot(
            kind="scatter",
            edgecolors='none',
            x=df.columns[0],
            y=df.columns[1],
            color=colors[i],
            label=label,
            ax=axis,
            zorder=df.shape[0] - df[label_mask].shape[0], # To put on top the smaller cluster
            alpha=0.5
        )
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(title)


def clusterize(X: pd.DataFrame, y: pd.DataFrame, mask: np.array, train_axes, test_axes, title: str):
    one, two = "First component", 'Second component'
    scaler = MinMaxScaler()
    std_mask= (np.abs(stats.zscore(X)) < X.shape[1]).all(axis=1)
    X, y, mask = X[std_mask], y[std_mask], mask[std_mask]
    X = pd.DataFrame(data=scaler.fit_transform(X), columns=[one, two])
    plot_clusters(X[mask], y[mask], train_axes, title.format(set_name="Train set"))
    plot_clusters(X[~mask], y[~mask], test_axes, title.format(set_name="Test set"))


def tsne(X: pd.DataFrame, y: pd.DataFrame, mask: np.array, train_axes, test_axes):
    clusterize(
        TSNE(n_jobs=cpu_count(), verbose=0, random_state=42).fit_transform(
            PCA(n_components=50, random_state=42).fit_transform(X)),
        y,
        mask,
        train_axes,
        test_axes,
        "{set_name} - TSNE for epigenomic data"
    )


def mca(X: pd.DataFrame, y: pd.DataFrame, mask: np.array, train_axes, test_axes):
    size = 50000
    idx = np.random.permutation(X.index.values)[:size]
    clusterize(
        MCA(X.iloc[idx]).fs_r(N=2),
        y.iloc[idx],
        mask[idx],
        train_axes,
        test_axes,
        "{set_name} - MCA for sequence data"
    )


def labelize(classes: np.ndarray, task: Dict) -> pd.DataFrame:
    labelized = pd.DataFrame(classes, columns=["labels"])
    labelized[labelized.labels == 1] = ", ".join(task["positive"])
    labelized[labelized.labels == 0] = ", ".join(task["negative"])
    return labelized


def reindex_nucleotides(X: np.ndarray, nucleotides=("a", "c", "g", "n", "t")):
    return pd.DataFrame(
        X.reshape(-1, X.shape[1]*X.shape[2]),
        columns=pd.MultiIndex.from_arrays(
            [nucleotides*X.shape[1], tuple(range(X.shape[1]))*X.shape[2]], names=['nucleotides', 'indices'])
    )


def visualize(target: str):
    with Notipy() as r:
        tasks = list(enumerate(tasks_generator(target)))
        for i, (target, cell_line, task, balance_mode) in tqdm(tasks):
            path = f"visualize/decompositions/{cell_line}"
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
            ((train_epigenomic, train_sequence, train_classes), (test_epigenomic,
                                                                    test_sequence, test_classes)), _, _ = next(generator())
            epigenomic = np.vstack([train_epigenomic, test_epigenomic])
            sequence = reindex_nucleotides(
                np.vstack([train_sequence, test_sequence]))
            classes = labelize(np.hstack([train_classes, test_classes]), task)
            mask = np.zeros(classes.size)
            mask[:train_classes.size] = 1
            mask = mask.astype(bool)
            _, axes = plt.subplots(1, 4, figsize=(6*4, 6))
            mca(sequence, classes, mask, axes[0], axes[1])
            tsne(epigenomic, classes, mask, axes[2], axes[3])
            plt.tight_layout()
            plt.savefig(f"{path}/{title}.png".replace(" ", "_"))
            plt.close()
            os.remove(f"{i}.tmp")
            r.add_report(pd.DataFrame({
                "cell line": cell_line,
                "task": task["name"],
                "balance_mode": balance_mode
            }, index=[i]))
