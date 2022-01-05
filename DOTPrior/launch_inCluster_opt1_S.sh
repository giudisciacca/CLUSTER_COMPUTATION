#!/bin/bash
#$ -l tmem=18G
#$ -l h_rt=10:00:00
#$ -S /bin/bash
#$ -j y
#$ -N python_launch1.sh
#$ -l gpu=true
#$ -t 34,35#1:49
#####  -pe gpu 1

# qrsh -l tmem=16G,gpu=true,h_rt=0:30:0 -pe gpu 2
#=========MATLAB===============
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.0.source

#export LD_LIBRARY_PATH="/share/apps/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export PATH=/share/apps/cuda-10.0/bin:/usr/local/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:/lib64:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH="/share/apps/python-3.7.2-shared/lib/:$LD_LIBRARY_PATH"
export PATH="$PATH:/share/apps/python-3.7.2-shared/bin/"
#export PATH="/share/apps/cuda-10.0/bin:$PATH"
export VENV_FOLDER="/home/gdisciac/Python72Venv10"

cd /home/gdisciac/DOTPrior/
#python3 -m venv $VENV_FOLDER
source $VENV_FOLDER/bin/activate
#pip install --upgrade pip
#pip install -r  requirements2.txt

# frmw 1 slice is shared
export x=($(python -c "import numpy;a,b=numpy.unravel_index($SGE_TASK_ID-1, (7,7), order='C'); print(a,b)"))
export ABS=${x[0]}
export SCA=${x[1]}
python DeepPrior_OpticalLoss_iter_SCAABS.py 'OpticalLoss_iter1_full_manyPRC' 1 1 'BORNsvdLarge2' 1 'USadded' $ABS $SCA ; 
rm /home/gdisciac/DOTPrior/tensorboard_prior/BORNsvdLarge2USadded/*/MatlabprcSCA${SCA}prcABS${ABS}*3UNET*binaryInput*
#python DeepPrior_OpticalLoss_iter_SCAABS.py 'OpticalLoss_iter1_full_manyPRC' 1 1 'FEMsvdLarge2' 1 'USadded' $ABS $SCA ;
#rm /home/gdisciac/DOTPrior/tensorboard_prior/FEMsvdLarge2USadded/*/MatlabprcSCA${SCA}prcABS${ABS}*3UNET*binaryInput*


echo end-script
