def _build_csv_path(target:str, directory:str, cell_line:str):
    return "{target}/{directory}/{cell_line}.csv".format(
        target=target,
        directory=directory,
        cell_line=cell_line
    )

def get_raw_epigenomic_data_path(target:str, cell_line:str):
    return _build_csv_path(target, "epigenomic_data", cell_line)

def get_raw_nucleotides_sequences_path(target:str, cell_line:str):
    return _build_csv_path(target, "one_hot_encoded_expanded_regions", cell_line)

def get_raw_classes_path(target:str, cell_line:str):
    return _build_csv_path(target, "one_hot_encoded_classes", cell_line)