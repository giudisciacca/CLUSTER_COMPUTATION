#!/bin/bash
#$ -l tmem=6G
#$ -l h_rt=08:00:00
#$ -S /bin/bash
#$ -j y
#$ -N diffnet
#$ -l gpu=true
#$ -l tscratch=30G
#$ -t 1:480 #149:160#21:32# 53:64#128:160
#####  -pe gpu 1

# qrsh -l tmem=16G,gpu=true,h_rt=0:30:0 -pe gpu 2
#=========MATLAB===============
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.0.source
export PATH=/share/apps/cuda-10.0/bin:/usr/local/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH="/share/apps/python-3.7.2-shared/lib/:$LD_LIBRARY_PATH"
export PATH="$PATH:/share/apps/python-3.7.2-shared/bin/"
export VENV_FOLDER="/home/gdisciac/Python72Venv10"

cd /home/gdisciac/CBH/
mkdir -p /scratch0/gdisciac/

source $VENV_FOLDER/bin/activate

cmd=$(sed -n ${SGE_TASK_ID}'{p;q}' FEMarray_of_python_DOCM.txt)
echo $cmd | bash
 
echo end-script
