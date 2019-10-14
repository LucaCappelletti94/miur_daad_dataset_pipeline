from typing import Dict, List, Tuple
from mca import MCA
import numpy as np
import time
import os
from humanize import naturaldate
from multiprocessing import cpu_count, Pool
from .load import tasks_generator
from miur_daad_balancing import get_callback
from .utils import load_cell_lines, load_raw_classes, load_raw_epigenomic_data, load_raw_nucleotides_sequences
from MulticoreTSNE import MulticoreTSNE as TSNE
from notipy_me import Notipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pylab
from matplotlib import pyplot as plt
from auto_tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from ddd_subplots import subplots, rotate
matplotlib.use('Agg')


def inverse_sigmoid(x):
    return 1/(1-np.exp(5*(x-0.5)))


def normalize_radius(n, min_n, max_n):
    if min_n == max_n:
        return 1
    return 1-(n-min_n)/(max_n - min_n)


def plot_clusters(*X: Tuple[np.ndarray], classes: pd.DataFrame, axis, title: str):
    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2'
    ]

    colors_map = {
        "A-E": 0,
        "A-P": 1,
        "A-X": 2,
        "I-E": 3,
        "I-P": 4,
        "I-X": 5,
        "UK": 6,
        "A-X, I-E, I-P, I-X, UK": 1,
        "A-E, A-P": 0
    }
    X = np.array(X).T
    chosen_colors = [
        colors[colors_map[c]] for c in classes.values.flatten()
    ] 
    axis.scatter(
        *X.T,
        # edgecolors='none',
        color=chosen_colors,
        s=0.1,
        # label=label,
        # To put on top the smaller cluster
        #zorder=total_elements - class_elements,
        # alpha=0.5
    )
    #     axis.get_legend().legendHandles[i].set_alpha(1)
    #     axis.get_legend().legendHandles[i]._sizes = [50]
    # axis.get_legend().set_title("Classes")
    # axis.get_legend().set_zorder(total_elements)
    # axis.set_xlim(-0.05, 1.05)
    # axis.set_ylim(-0.05, 1.05)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.set_title(title)


def clusterize(X: pd.DataFrame, y: pd.DataFrame, masks: List[np.array], axes: List, titles: List[str]):
    for _ in range(2):
        std_mask = (np.abs(stats.zscore(X)) < 3).all(axis=1)
        X, y, masks = X[std_mask], y[std_mask], [
            mask[std_mask] for mask in masks
        ]
    X = pd.DataFrame(data=MinMaxScaler().fit_transform(
        X), columns=["First component", 'Second component'])
    for mask, ax, title in zip(masks, axes, titles):
        plot_clusters(
            *X[mask].values.T,
            classes=y[mask],
            axis=ax,
            title=title
        )


def clusterize_3d(*X: Tuple[np.ndarray], classes: pd.DataFrame, masks: List[np.array], subplots_args=None, axes=None, titles: List[str]):
    #fig, axes = subplots(**subplots_args, subplot_kw={"zscale":"linear"})
    axes = axes.flatten()
    X = np.array(X).T
    for mask, ax, title in zip(masks, axes, titles):
        plot_clusters(
            *X[mask].T,
            classes=classes[mask],
            axis=ax,
            title=title
        )
    #return fig, axes


def tsne(X: pd.DataFrame, dimensions: int):
    return pd.DataFrame(TSNE(n_jobs=cpu_count(), verbose=0, random_state=42, n_components=dimensions).fit_transform(
        PCA(n_components=50, random_state=42).fit_transform(
            X) if X.shape[1] > 50 else X
    ), index=X.index)


def mca(X: pd.DataFrame, dimensions: int):
    size = 50000
    idx = np.random.permutation(X.index.values)
    X = X.reindex(idx)
    return pd.concat([
        pd.DataFrame(data=MinMaxScaler().fit_transform(MCA(
            X.iloc[i:i+size]
        ).fs_r(N=dimensions)), index=X.iloc[i:i+size].index) for i in range(0, len(X), size)
    ]).reindex(sorted(X.index.values))


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


def build_cell_line_visualization(classes: pd.DataFrame, title: str, n: int = 4):
    uniques = sorted(set(classes.values.flatten()))
    h = np.ceil((len(uniques)+1)/n).astype(int)
    _, axes = plt.subplots(h, n, figsize=(4*n, 4*h))
    flat_axes = np.array(axes).flatten()
    for axis in flat_axes[len(uniques)+1:]:
        axis.axis("off")
    titles = [
        f"{c}" for c in uniques
    ]
    titles.append(title)
    masks = []
    for u in uniques:
        mask = np.zeros(classes.size)
        mask[classes.values.flatten() == u] = 1
        masks.append(mask.astype(bool))
    masks.append(np.ones(classes.size).astype(bool))
    return masks, flat_axes, titles


def build_cell_line_visualization_3d(classes: pd.DataFrame, title: str, n: int = 4):
    masks, _, titles = build_cell_line_visualization(classes, title, n)
    uniques = sorted(set(classes.values.flatten()))
    h = np.ceil((len(uniques)+1)/n).astype(int)
    return masks, {"ncols": n, "nrows": h, "figsize": (2.5*n, 2.5*h)}, titles


def build_cell_line_big_visualization(classes: pd.DataFrame, title: str):
    _, axes = plt.subplots(1, 1, figsize=(7, 7))
    return [np.ones(classes.size).astype(bool)], [axes], [title]


def build_cell_line_big_visualization_3d(classes: pd.DataFrame, title: str):
    masks, _, titles = build_cell_line_big_visualization(classes, title)
    return masks, {"ncols": 1, "nrows": 1, "figsize": (7, 7)}, titles


def visualize_cell_lines_nucleotides(target: str, cell_line: str, path: str, classes: pd.DataFrame, args):
    build_cache(path)
    sequence = pd.read_csv(f"{cell_line}-mca.csv", index_col=[0]).values
    clusterize(sequence, classes, *args)
    save_pic(path)
    clear_cache(path)


def visualize_cell_lines_nucleotides_3d(target: str, cell_line: str, path: str, classes: pd.DataFrame, masks: List, subplots_args: Dict, titles: List):
    build_cache(path)
    sequence = pd.read_csv(f"{cell_line}-mca3.csv", index_col=[0]).values
    unknown_mask = (classes.values != "UK").flatten()
    masks = [
        mask[unknown_mask] for mask in masks
    ]
    sequence = sequence[unknown_mask]
    classes = classes[unknown_mask]
    for _ in range(2):
        std_mask = (np.abs(stats.zscore(sequence)) < 3).all(axis=1)
        sequence, classes, masks = sequence[std_mask], classes[std_mask], [
            mask[std_mask] for mask in masks
        ]
    rotate(clusterize_3d, *sequence.T, duration=3, path=path, parallelize=True, verbose=True,
           classes=classes, masks=masks, subplots_args=subplots_args, titles=titles)
    clear_cache(path)


def visualize_cell_lines_epigenomic(target: str, cell_line: str, path: str, classes: pd.DataFrame, args):
    build_cache(path)
    epigenomic = pd.read_csv(f"{cell_line}-tsne.csv", index_col=[0]).values
    clusterize(epigenomic, classes, *args)
    save_pic(path)
    clear_cache(path)


def visualize_cell_lines_epigenomic_3d(target: str, cell_line: str, path: str, classes: pd.DataFrame, masks: List, subplots_args: Dict, titles: List):
    build_cache(path)
    epigenomic = pd.read_csv(f"{cell_line}-tsne3.csv", index_col=[0]).values
    rotate(clusterize_3d, *epigenomic.T, duration=3, path=path, parallelize=True,
           verbose=True, classes=classes, masks=masks, subplots_args=subplots_args, titles=titles)
    clear_cache(path)


def visualize_cell_lines_job(job: Tuple):
    target, cell_line = job
    path = f"visualize/decompositions/{cell_line}"
    os.makedirs(path, exist_ok=True)
    paths = [
        f"{path}/mca.png",
        f"{path}/tsne.png",
        f"{path}/big-mca.png",
        f"{path}/big-tsne.png",
        f"{path}/big-tsne3.gif",
        f"{path}/tsne3.gif",
        f"{path}/mca3.gif",
    ]
    if all([not can_run(p) for p in paths]):
        return
    classes = load_raw_classes(target, cell_line)
    # if can_run(paths[0]):
    #     visualize_cell_lines_nucleotides(
    #         target, cell_line, paths[0], classes, build_cell_line_visualization(classes, "MCA of sequence data"))
    # if can_run(paths[1]):
    #     visualize_cell_lines_epigenomic(target, cell_line, paths[1], classes, build_cell_line_visualization(
    #         classes, "TSNE of epigenomic data"))
    # if can_run(paths[2]):
    #     visualize_cell_lines_nucleotides(
    #         target, cell_line, paths[2], classes, build_cell_line_big_visualization(classes, "MCA of sequence data"))
    # if can_run(paths[3]):
    #     visualize_cell_lines_epigenomic(target, cell_line, paths[3], classes, build_cell_line_big_visualization(
    #         classes, "TSNE of epigenomic data"))
    # if can_run(paths[4]):
    #     visualize_cell_lines_epigenomic_3d(
    #         target, cell_line, paths[4], classes, *build_cell_line_big_visualization_3d(classes, "TSNE of epigenomic data"))
    if can_run(paths[5]):
        visualize_cell_lines_epigenomic_3d(
            target, cell_line, paths[5], classes, *build_cell_line_visualization_3d(classes, "TSNE of epigenomic data"))
    if can_run(paths[6]):
        visualize_cell_lines_nucleotides_3d(
            target, cell_line, paths[6], classes, *build_cell_line_visualization_3d(classes, "MCA of epigenomic data"))


def visualize_cell_lines(target: str):
    jobs = [
        (target, cell_line) for cell_line in load_cell_lines(target)
    ]
    # with Pool(cpu_count()) as p:
    #     list(tqdm(p.imap(visualize_cell_lines_job, jobs),
    #               total=len(jobs), desc="Visualizing tasks"))
    for job in tqdm(jobs[:1], desc="Visualizing tasks"):
        visualize_cell_lines_job(job)


def visualize_tasks_job(job: Tuple):
    target, cell_line, task, balance_mode = job
    path = "visualize/decompositions/{cell_line}/{cell_line}-{balance_mode}-{task}.png".format(
        task=task["name"],
        cell_line=cell_line,
        balance_mode=balance_mode.replace("umbalanced", "unbalanced")
    ).replace(" ", "_")
    # if not (can_run(cell_line) and can_run(path)):
    #    return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    build_cache(path)
    classes = load_raw_classes(target, cell_line).values

    used_classes = task["positive"] + task["negative"]
    mask = np.array([
        c in used_classes for c in classes
    ])

    epigenomic = pd.read_csv(
        f"{cell_line}-tsne3.csv", index_col=[0]).values[mask]
    sequence = pd.read_csv(f"{cell_line}-mca3.csv", index_col=[0]).values[mask]
    classes = classes[mask]

    balancer = get_callback(balance_mode)
    epigenomic_train, epigenomic_test, sequence_train, sequence_test, classes_train, classes_test = train_test_split(
        epigenomic, sequence, classes, test_size=0.3, random_state=42)

    (epigenomic_train, sequence_train, classes_train), (epigenomic_test, sequence_test, classes_test) = balancer(
        (epigenomic_train, sequence_train, classes_train),
        (epigenomic_test, sequence_test, classes_test)
    )

    classes = labelize(np.array([
        int(c in task["positive"]) for c in np.vstack([classes_train, classes_test])
    ]), task)

    epigenomic = np.vstack([epigenomic_train, epigenomic_test])
    sequence = np.vstack([sequence_train, sequence_test])
    mask = np.zeros(classes.size)
    mask[: classes_train.size] = 1
    mask = mask.astype(bool)
    _, axes = subplots(1, 4, figsize=(4*4, 4))
    axes = axes.flatten()
    masks = [mask, ~mask]

    clusterize_3d(*sequence.T, classes=classes, masks=masks, axes=axes[: 2], titles=[
        "MCA - Sequence data - Train set",
        "MCA - Sequence data - Test set"
    ])
    clusterize_3d(*epigenomic.T, classes=classes, masks=masks, axes=axes[2:], titles=[
        "TSNE - Epigenomic data - Train set",
        "TSNE - Epigenomic data - Test set"
    ])
    save_pic(path)
    clear_cache(path)


def visualize_tasks(target: str):
    tasks = list(tasks_generator(target))
    # with Pool(cpu_count()) as p:
    #     list(tqdm(p.imap(visualize_tasks_job, tasks),
    #               total=len(tasks), desc="Visualizing tasks"))
    for task in tqdm(tasks):
        visualize_tasks_job(task)


def build_tsne(target: str):
    for cell_line in tqdm(load_cell_lines(target), desc="Buiding epigenomic data TSNE."):
        path = f"{cell_line}-tsne.csv"
        if not can_run(path):
            continue
        build_cache(path)
        tsne(load_raw_epigenomic_data(target, cell_line), 2).to_csv(path)
        clear_cache(path)

    for cell_line in tqdm(load_cell_lines(target), desc="Buiding epigenomic data TSNE."):
        path = f"{cell_line}-tsne3.csv"
        if not can_run(path):
            continue
        build_cache(path)
        tsne(load_raw_epigenomic_data(target, cell_line), 3).to_csv(path)
        clear_cache(path)


def build_mca(target: str):
    for cell_line in tqdm(load_cell_lines(target), desc="Buiding epigenomic data MCA."):
        path = f"{cell_line}-mca.csv"
        if not can_run(path):
            continue
        build_cache(path)
        mca(reindex_nucleotides(
            load_raw_nucleotides_sequences(target, cell_line)), 2
            ).to_csv(path)
        clear_cache(path)

    for cell_line in tqdm(load_cell_lines(target), desc="Buiding epigenomic data MCA."):
        path = f"{cell_line}-mca3.csv"
        if not can_run(path):
            continue
        build_cache(path)
        mca(reindex_nucleotides(
            load_raw_nucleotides_sequences(target, cell_line)), 3
            ).to_csv(path)
        clear_cache(path)


def visualize(target: str):
    matplotlib.rcParams['figure.dpi'] = 70
    # with Notipy():
    #     build_mca(target)
    #     build_tsne(target)
    visualize_tasks(target)
    #visualize_cell_lines(target)
