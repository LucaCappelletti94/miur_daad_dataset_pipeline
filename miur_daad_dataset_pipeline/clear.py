import shutil
import os
from holdouts_generator import clear_cache


def clear(target: str):
    dirs = (
        "one_hot_encoded_classes",
        "one_hot_encoded_expanded_regions",
        "cell_lines"
    )
    for d in dirs:
        shutil.rmtree(
            "{target}/{d}".format(target=target, d=d), ignore_errors=True
        )

    gzip_dir = "{target}/epigenomic_data".format(target=target)
    for document in os.listdir(gzip_dir):
        if document.endswith(".csv"):
            os.remove(
                "{gzip_dir}/{document}".format(gzip_dir=gzip_dir,
                                               document=document)
            )

    clear_cache(".holdouts")