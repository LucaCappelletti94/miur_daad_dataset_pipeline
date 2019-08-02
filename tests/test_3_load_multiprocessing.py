from miur_daad_dataset_pipeline import task_builder
from miur_daad_dataset_pipeline.utils import load_gaussian_process_holdouts
from auto_tqdm import tqdm
from holdouts_generator import clear_cache

def test_load_multiprocessing():
    target = "test_dataset"
    task_builder(target, load_gaussian_process_holdouts(target))
    clear_cache()