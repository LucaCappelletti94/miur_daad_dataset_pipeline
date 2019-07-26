import gzip
import shutil

def ungzip(path:str):
    with gzip.open(path, 'rb') as f_in:
        with open(path.strip(".gz"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)