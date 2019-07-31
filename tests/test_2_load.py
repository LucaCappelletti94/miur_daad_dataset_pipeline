from miur_daad_dataset_pipeline import tasks_generator, balanced_holdouts_generator
from miur_daad_dataset_pipeline.utils import load_holdouts
from miur_daad_dataset_pipeline.utils import load_gaussian_process_holdouts
from auto_tqdm import tqdm

def test_load():
    target = "test_dataset"
    for task in tqdm(list(tasks_generator(target)), desc="Jobs"):
        for _, _, sub in balanced_holdouts_generator(*task, load_gaussian_process_holdouts(target))():
            for _ in sub():
                pass
        for _, _, sub in balanced_holdouts_generator(*task, load_holdouts(target))():
            for _ in sub():
                pass