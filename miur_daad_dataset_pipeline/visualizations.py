from typing import Dict, List
from mca import MCA
import numpy as np
import time
import os
from humanize import naturaldate
from multiprocessing import cpu_count
from .load import tasks_generator, balanced_holdouts_generator
from .utils import load_cell_lines, load_raw_classes, load_raw_epigenomic_data, load_raw_nucleotides_sequences
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
            # To put on top the smaller cluster
            zorder=df.shape[0] - df[label_mask].shape[0],
            alpha=0.5
        )
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(title)


def clusterize(X: pd.DataFrame, y: pd.DataFrame, masks: List[np.array], axes: List, titles: List[str]):
    for _ in range(2):
        std_mask = (np.abs(stats.zscore(X)) < 3).all(axis=1)
        X, y, masks = X[std_mask], y[std_mask], [
            mask[std_mask] for mask in masks
        ]
    X = pd.DataFrame(data=MinMaxScaler().fit_transform(
        X), columns=["First component", 'Second component'])
    for mask, ax, title in zip(mask, axes, titles):
        plot_clusters(X[mask], y[mask], ax, title)


def tsne(X: pd.DataFrame, y: pd.DataFrame, masks: List[np.array], axes: List, titles: List[str]):
    clusterize(
        TSNE(n_jobs=cpu_count(), verbose=0, random_state=42).fit_transform(
            PCA(n_components=50, random_state=42).fit_transform(X)),
        y,
        masks,
        axes,
        titles
    )


def mca(X: pd.DataFrame, y: pd.DataFrame, masks: List[np.array], axes: List, titles: List[str]):
    size = 50000
    idx = np.random.permutation(X.index.values)
    X = X.iloc[idx]
    clusterize(
        pd.concat([
            pd.DataFrame(data=MinMaxScaler().fit_transform(MCA(
                X.loc[i:i+size-1, :]
            ).fs_r(N=2)), columns=[1, 2]) for i in range(0, len(X), size)
        ]),
        y.iloc[idx],
        [m[idx] for m in masks],
        axes,
        titles
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


def save_pic(path: str):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def can_run(path: str):
    return not any([os.path.exists(p) for p in [path, f"{path}.tmp"]])


def build_cache(path: str):
    with open(f"{path}.tmp", "w") as f:
        f.write("")


def clear_cache(path: str):
    if os.path.exists(f"{path}.tmp"):
        os.remove(f"{path}.tmp")


def visualize_cell_lines_nucleotides(target: str, cell_line: str, path: str, classes: pd.DataFrame):
    build_cache(path)
    sequence = reindex_nucleotides(
        load_raw_nucleotides_sequences(target, cell_line))
    _, axes = plt.subplots(1, 1, figsize=(30, 30))
    mca(
        sequence,
        classes,
        [np.ones(classes.size).astype(bool)],
        axes,
        ["MCA decomposition of sequence data"]
    )
    save_pic(path)
    clear_cache(path)


def visualize_cell_lines_epigenomic(target: str, cell_line: str, path: str, classes: pd.DataFrame):
    build_cache(path)
    epigenomic = load_raw_epigenomic_data(target, cell_line)
    _, axes = plt.subplots(1, 1, figsize=(30, 30))
    tsne(
        epigenomic,
        classes,
        [np.ones(classes.size).astype(bool)],
        axes,
        ["TSNE decomposition of epigenomic data"]
    )
    save_pic(path)
    clear_cache(path)


def visualize_cell_lines_mixed(target: str, cell_line: str, path: str, classes: pd.DataFrame):
    build_cache(path)
    epigenomic = load_raw_epigenomic_data(target, cell_line)
    sequence = reindex_nucleotides(load_raw_nucleotides_sequences(target, cell_line))
    _, axes = plt.subplots(1, 1, figsize=(30, 30))
    size = 50000
    idx = np.random.permutation(epigenomic.index.values)
    epigenomic = epigenomic.iloc[idx]
    sequence = sequence.iloc[idx]
    classes = classes.iloc[idx]
    tsne(
        pd.concat([
            pd.concat([
                pd.DataFrame(data=MinMaxScaler().fit_transform(MCA(
                    sequence.loc[i:i+size-1, :]
                ).fs_r())) for i in range(0, len(sequence), size)
            ]), epigenomic],
            axis=1
        ),
        classes,
        [np.ones(classes.size).astype(bool)],
        axes,
        ["TSNE decomposition of mixed data"]
    )
    save_pic(path)
    clear_cache(path)


def visualize_cell_lines(target: str):
    for cell_line in tqdm(load_cell_lines(target)):
        path = f"visualize/decompositions/{cell_line}"
        os.makedirs(path, exist_ok=True)
        paths = [
            f"{path}/mca.png",
            f"{path}/tsne.png",
            f"{path}/mixed-tsne.png"
        ]
        if all([not can_run(p) for p in paths]):
            continue
        classes = load_raw_classes(target, cell_line)
        if can_run(paths[0]):
            visualize_cell_lines_nucleotides(
                target, cell_line, paths[0], classes)
        if can_run(paths[1]):
            visualize_cell_lines_epigenomic(
                target, cell_line, paths[1], classes)
        if can_run(paths[2]):
            visualize_cell_lines_mixed(
                target, cell_line, paths[2], classes)


def visualize_tasks(target: str):
    for target, cell_line, task, balance_mode in tasks_generator(target):
        path = "visualize/decompositions/{cell_line}/{cell_line}-{balance_mode}-{task}.png".format(
            task=task["name"],
            cell_line=cell_line,
            balance_mode=balance_mode.replace("umbalanced", "unbalanced")
        ).replace(" ", "_")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if can_run(path):
            build_cache(path)
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
            masks = [mask, ~mask]

            mca(sequence, classes, masks, axes[:2], [
                "MCA decomposition of sequence data train set",
                "MCA decomposition of sequence data test set"
            ])
            tsne(epigenomic, classes, masks, axes[2:], [
                "TSNE decomposition of epigenomic data train set",
                "TSNE decomposition of epigenomic data test set"
            ])
            save_pic(path)
            clear_cache(path)


def visualize(target: str):
    with Notipy():
        # visualize_tasks(target)
        visualize_cell_lines(target)
