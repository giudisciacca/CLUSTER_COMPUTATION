#!/bin/bash
#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=48:00:00
#$ -S /bin/bash
#$ -j y
#$ -N python_launch.sh
#$ -l gpu=true
#####  -pe gpu 2

# qrsh -l tmem=16G,gpu=true,h_rt=0:30:0 -pe gpu 2
#=========MATLAB===============
source /share/apps/source_files/python/python-3.6.4.source
source /share/apps/source_files/cuda/cuda-9.0.source

export PATH="/home/gdisciac/.local/lib/python3.6/site-packages/:$PATH:/home/gdisciac/.local/lib/python3.6/site-packages/"
export PATH="/home/gdisciac/.local/bin:$PATH:/home/gdisciac/.local/bin"
export PATH="/share/apps/matlabR2018b/bin:$PATH"
export PATH="/home/gdisciac/mcx/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/share/apps/python-3.6.4-shared/lib/:/home/gdisciac/.local/lib/:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/apps/python-3.6.4-shared/lib/"
export PATH="$PATH:/share/apps/python-3.6.4-shared/bin/"
export PATH="/share/apps/cuda-9.0/bin:$PATH"
#export PATH="$PATH:/share/apps/cuda-9.0"
#export PATH="/home/gdisciac/mcx/bin:$PATH"
export VENV_FOLDER="/home/gdisciac/PythonVenv9"

cd /home/gdisciac/DOTPrior/
python3 -m venv $VENV_FOLDER
source $VENV_FOLDER/bin/activate
pip install --upgrade pip
pip install -r  requirements.txt

python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_full' 1 1 'BORNsvd' 3;
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_optic' 1 0 'BORNsvd' 3;
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_slice' 0 1 'BORNsvd' 3;


python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full' 1 1 'BORNsvd' 1;
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic' 1 0 'BORNsvd' 1;
python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice' 0 1 'BORNsvd' 1;


#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full' 1 1 'BORNsvdNoNorm' 1;python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic' 1 0 'BORNsvdNoNorm' 1;python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice' 0 1 'BORNsvdNoNorm' 1;python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_full' 1 1 'BORNsvdNoNorm' 3;python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_optic' 1 0 'BORNsvdNoNorm' 3;python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter3_slice' 0 1 'BORNsvdNoNorm' 3;
echo end-script
