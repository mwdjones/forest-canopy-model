#!/bin/bash
#SBATCH --job-name=jonesPyMesh
#SBATCH --output jonesPyMesh_%j.output  
#SBATCH --error jonesPyMesh_%j.error
#SBATCH --nodes=1  
#SBATCH --ntasks=1 

module load conda
conda activate pymesh

python 02_run-simulations3D.py
