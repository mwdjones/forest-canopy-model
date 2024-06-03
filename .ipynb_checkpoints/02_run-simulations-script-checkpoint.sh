#!/bin/bash -l
#SBATCH --job-name=jonesPyMesh
#SBATCH --output jonesPyMesh_%j.output  
#SBATCH --error jonesPyMesh_%j.error
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=200M

module load conda
conda init bash
conda activate pymesh

python 02_run-simulations3D.py
