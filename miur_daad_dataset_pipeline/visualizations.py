import matplotlib
matplotlib.use('Agg') 
import pandas as pd
from auto_tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from notipy_me import Notipy
from MulticoreTSNE import MulticoreTSNE as TSNE
from .utils import load_raw_epigenomic_data, load_cell_lines, load_raw_classes
from multiprocessing import cpu_count
from humanize import naturaldate
import os
import time

def heatmap(df:pd.DataFrame, cell_line:str):
    plt.rcParams['figure.figsize'] = [80, 80]
    plt.axis('scaled')
    sns.heatmap(
        df.corr(), 
        annot=True,
        xticklabels=df.columns,
        yticklabels=df.columns,
        cbar=False
    )
    title= f"Correlation map for cell line {cell_line}"
    path = "visualize/correlation"
    os.makedirs(path, exist_ok=True)
    plt.title(title)
    plt.savefig(f"{path}/{title}.png")

def scatter(df:pd.DataFrame, cell_line:str):
    pd.plotting.scatter_matrix(df, alpha = 0.2, figsize = (200, 200), diagonal = 'kde')
    title= f"Scatter plot for cell line {cell_line}"
    path = "visualize/scatter"
    os.makedirs(path)
    plt.title(title)
    plt.savefig(f"{path}/{title}.png")

classes_colors = {
    "I-X":"red",
    "A-P":"blue",
    "I-P":"green",
    "A-X":"yellow",
    "UK":"black",
    "I-E":"cyan",
    "A-E":"magenta"
}

def clustering(df:pd.DataFrame, classes:pd.DataFrame, cell_line:str, method, method_name:str):
    plt.rcParams['figure.figsize'] = [8, 8]
    one, two = "First component", 'Second component'
    reduction = pd.DataFrame(data=method.fit_transform(df), columns=[one, two])
    mask = (reduction.abs() < 2*reduction.std()).any(axis=1)
    reduction, classes = reduction[mask], classes[mask]
    reduction = (reduction-reduction.min())/(reduction.max()-reduction.min())
    clustered = pd.concat([reduction, classes], axis=1)
    ax = None
    for cls in set(clustered.labels.values):
        mask = clustered.labels == cls
        if ax is None:
            ax = clustered[mask].plot(kind="scatter", x=one,y=two, color=classes_colors[cls], label=cls, alpha=0.6)
        else:
            clustered[mask].plot(kind="scatter", x=one,y=two, color=classes_colors[cls], label=cls, ax=ax, alpha=0.6)
    title = f"{method_name} reduction for {cell_line} epigenomic data."
    path = "visualize/clustering"
    os.makedirs(path, exist_ok=True)
    plt.title(title)
    plt.savefig(f"{path}/{title}.png")
    plt.close()

def pca(df:pd.DataFrame, classes:pd.DataFrame, cell_line:str):
    pca = PCA(n_components=2)
    clustering(df, classes, cell_line, pca, "PCA")

def tsne(df:pd.DataFrame, classes:pd.DataFrame, cell_line:str):
    tsne = TSNE(n_jobs=cpu_count(), verbose=2)
    clustering(df, classes, cell_line, tsne, "TSNE")


def visualize(target:str, cell_line:str):
    with Notipy() as r:
        start = time.time()
        df = load_raw_epigenomic_data(target, cell_line)
        classes = load_raw_classes(target, cell_line)
        classes.columns = ["labels"]
        pca(df, classes, cell_line)
        tsne(df, classes, cell_line)
        heatmap(df, cell_line)
        scatter(df, cell_line)
        r.add_report(pd.DataFrame({
            "cell line":cell_line,
            "time":naturaldate(time.time() - start)
        }))