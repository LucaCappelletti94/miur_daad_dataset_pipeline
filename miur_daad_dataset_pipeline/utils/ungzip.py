import gzip
import shutil

def ungzip(path:str):
    """Extract given file gz to same path.
        path:str, path to file gz to extract.
    """
    with gzip.open(path, 'rb') as f_in:
        with open(path.split(".gz")[0], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)