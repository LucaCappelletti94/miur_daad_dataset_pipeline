import matplotlib
matplotlib.use('Agg')
from typing import Dict
import numpy as np
import time
import os
from humanize import naturaldate
from multiprocessing import cpu_count
from .load import tasks_generator, balanced_holdouts_generator
#from MulticoreTSNE import MulticoreTSNE as TSNE
from notipy_me import Notipy
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from auto_tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


# def heatmap(df: pd.DataFrame, cell_line: str):
#     plt.rcParams['figure.figsize'] = [80, 80]
#     plt.axis('scaled')
#     sns.heatmap(
#         df.corr(),
#         annot=True,
#         xticklabels=df.columns,
#         yticklabels=df.columns,
#         cbar=False
#     )
#     title = f"Correlation map for cell line {cell_line}"
#     path = "visualize/correlation"
#     os.makedirs(path, exist_ok=True)
#     plt.title(title)
#     plt.savefig(f"{path}/{title}.png")


# def scatter(df: pd.DataFrame, cell_line: str):
#     pd.plotting.scatter_matrix(
#         df, alpha=0.2, figsize=(200, 200), diagonal='kde')
#     title = f"Scatter plot for cell line {cell_line}"
#     path = "visualize/scatter"
#     os.makedirs(path)
#     plt.title(title)
#     plt.savefig(f"{path}/{title}.png")


classes_colors = {
    "I-X": "red",
    "A-P": "blue",
    "I-P": "green",
    "A-X": "yellow",
    "UK": "black",
    "I-E": "cyan",
    "A-E": "magenta"
}


def clustering(df: pd.DataFrame, classes: pd.DataFrame, cell_line: str, method, method_name: str, axis, set_name: str):
    plt.rcParams['figure.figsize'] = [8, 8]
    one, two = "First component", 'Second component'
    reduction = pd.DataFrame(data=method.fit_transform(df), columns=[one, two])
    mask = (reduction.abs() < 2*reduction.std()).any(axis=1)
    reduction, classes = reduction[mask], classes[mask]
    reduction = (reduction-reduction.min())/(reduction.max()-reduction.min())
    clustered = pd.concat([reduction, classes], axis=1)
    for cls in set(clustered.labels.values):
        mask = clustered.labels == cls
        clustered[mask].plot(kind="scatter", x=one, y=two,
                             color=classes_colors[cls], label=cls, ax=axis, alpha=0.4)
    axis.set_title(f"{method_name} reduction of {set_name}")

def pca(df: pd.DataFrame, classes: pd.DataFrame, cell_line: str, axis, set_name: str):
    pca = PCA(n_components=2)
    clustering(df, classes, cell_line, pca, "PCA", axis, set_name)


def tsne(df: pd.DataFrame, classes: pd.DataFrame, cell_line: str, axis, set_name: str):
    tsne = TSNE(n_jobs=cpu_count(), verbose=2)
    clustering(df, classes, cell_line, tsne, "TSNE", axis, set_name)


def labelize(classes: np.ndarray, task: Dict) -> pd.DataFrame:
    labelized = pd.DataFrame(classes, columns=["labels"])
    labelized[labelized.labels == 1] = ", ".join(task["positive"])
    labelized[labelized.labels == 0] = ", ".join(task["negative"])
    return labelized


def visualize(target: str):
    with Notipy() as r:
        os.makedirs("visualize", exist_ok=True)
        tasks = list(tasks_generator(target))
        for target, cell_line, task, balance_mode in tqdm(tasks):
            generator = balanced_holdouts_generator(target, cell_line, task, balance_mode, {
                "quantities": [1],
                "test_sizes": [0.3]
            }, verbose=False)
            ((train_x, train_y), (test_x, test_y)), _, _ = next(generator())
            train_y = labelize(train_y, task)
            test_y = labelize(test_y, task)
            _, axes = plt.subplots(1, 4, figsize=(8*4, 8))
            pca(train_x, train_y, cell_line, axes[0], "training set")
            pca(test_x, test_y, cell_line, axes[1], "testing set")
            tsne(train_x, train_y, cell_line, axes[2], "training set")
            tsne(test_x, test_y, cell_line, axes[3], "testing set")
            title = "{cell_line} - {task} - {balance_mode}".format(
                task=task["name"],
                cell_line=cell_line,
                balance_mode=balance_mode
            )
            plt.tight_layout()
            plt.savefig(f"visualize/{title}.png")
            plt.close()
            r.add_report(pd.DataFrame({
                "cell line": cell_line,
                "task": task["name"],
                "balance_mode": balance_mode
            }))
