#!/bin/bash
#SBATCH --account=ELIX4_valentin
#SBATCH --partition=gll_usr_prod
#SBATCH --cpus-per-task=36
#SBATCH --mem=100GB
#SBATCH --time 24:00:00
#SBATCH --job-name=CNN-CPU-VISUALIZATION
BASEDIR="/gpfs/work/ELIX4_valentin/mlp_vs_cnn/miur_daad_dataset_pipeline"
VENVPYTHONDIR="/gpfs/work/ELIX4_valentin/mlp_vs_cnn/mlp_vs_cnn_venv"
cd $BASEDIR
module load python/3.6.4
source $VENVPYTHONDIR/bin/activate
python3 -c 'from miur_daad_dataset_pipeline import visualize; visualize("dataset")'
deactivate
