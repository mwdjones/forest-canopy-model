#!/bin/bash
#PBS -N treemodel3d
#PBS -A UMIN0008
#PBS -l walltime=12:00:00
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1:mem=10GB
#PBS -m abe
#PBS -M jone3247@umn.edu

module load conda
conda activate pymesh

python 02_run-simulations3D.py
