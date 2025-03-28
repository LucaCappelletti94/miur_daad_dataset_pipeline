import os
import re
# To use a consistent encoding
from codecs import open as copen
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with copen(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("miur_daad_dataset_pipeline", "__version__.py")

test_deps = ['pytest', 'pytest-cov', 'coveralls', 'validate_version_code', 'codacy-coverage']

extras = {
    'test': test_deps,
}

setup(
    name='miur_daad_dataset_pipeline',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='Simple python package to render the holdouts and training datasets of active regulatory regions for models with the task to predict them.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/LucaCappelletti94/miur_daad_dataset_pipeline',

    # Author details
    author='Luca Cappelletti',
    author_email='cappelletti.luca94@gmail.com',

    # Choose your license
    license='MIT',

    include_package_data=True,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
    ],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    tests_require=test_deps,
    install_requires=[
        'miur_daad_balancing',
        'fasta_one_hot_encoder',
        'pandas',
        'auto_tqdm',
        'holdouts_generator',
        'ucsc_genomes_downloader',
        'notipy_me',
        'silence_tensorflow',
        'extra_keras_metrics',
        'extra_keras_utils',
        'gaussian_process',
        'keras_tqdm',
        "matplotlib",
        "seaborn",
        "cmake",
        "mca",
        "ddd_subplots",
        "MulticoreTSNE"
    ],
    extras_require=extras,
)
