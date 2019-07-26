from .utils import expand_regions, one_hot_encode_classes, one_hot_encode_expanded_regions, ungzip_data, load_settings, sanitize
from ucsc_genomes_downloader import download_genome

def build(target:str):
    settings = load_settings(target)
    genome = settings["genome"]
    download_genome(genome, path=target)
    ungzip_data(target, settings)
    expand_regions(target, genome, settings)
    one_hot_encode_classes(target, settings)
    one_hot_encode_expanded_regions(target, settings)
    sanitize(target, settings)