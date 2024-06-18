#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J Sweep
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output_%J.out 
#BSUB -ee Output_%J.err 

# load a module
# replace VERSION 
module load python3/3.11.3

# load CUDA (for GPU support)
# load the correct CUDA for the pytorch version you have installed
module load cuda/12.1

# activate the virtual environment
# NOTE: needs to have been built with the same numpy / SciPy  version as above!
source fleetenv/bin/activate

wandb agent denniscz/CrackDetect-src/wr3unm13