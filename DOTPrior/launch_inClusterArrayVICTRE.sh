#!/bin/bash
#$ -l tmem=25G
#$ -l h_rt=18:00:00
#$ -S /bin/bash
#$ -j y
#$ -N pyArray
#$ -l gpu=true
#$ -t 1:50
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

cd /home/gdisciac/DOTPrior/
#python3 -m venv $VENV_FOLDER
source $VENV_FOLDER/bin/activate
#pip install --upgrade pip
#pip install -r  requirements2.txt

#export TF_FOLDER=/scratch0/gdisciac/tensorboard_prior/
#mkdir $TF_FOLDER
cmd=$(sed -n ${SGE_TASK_ID}'{p;q}' array_of_python_victre_multilearning.txt)
echo $cmd | bash

DEL_VAR_TOT=($cmd)
DEL_VAR_METHOD=${DEL_VAR_TOT[2]}
DEL_VAR_METHOD=$(echo $DEL_VAR_METHOD | tr -d \')
DEL_VAR_DATA=${DEL_VAR_TOT[5]}${DEL_VAR_TOT[7]} 
DEL_VAR_DATA=$(echo $DEL_VAR_DATA | tr -d \')
ASSEMBLED=/home/gdisciac/DOTPrior/tensorboard_prior/$DEL_VAR_DATA/$DEL_VAR_METHOD/*best_binaryInput*
rm $ASSEMBLED


####$cmd
#rm /home/gdisciac/DOTPrior/tensorboard_prior/'+ sys.argv[4] +sys.argv[6]+ '/' + sys.argv[1] + '/'
#tensorboard_dir = '/scratch0/gdisciac/tensorboard_prior/' + sys.argv[4] +sys.argv[6]+ '/' + sys.argv[1] + '/'
#cp tensorboard_dir 
echo end-script
