#!/bin/bash
#$ -l tmem=5G
#$ -l h_vmem=60G
#$ -l h_rt=36:00:00
#$ -S /bin/bash
#$ -j y
#$ -N python_launch12.sh
#$ -l gpu=true,gpu_titanx=yes
#####  -pe gpu 1

# qrsh -l tmem=16G,gpu=true,h_rt=0:30:0 -pe gpu 2
#=========MATLAB===============
source /share/apps/source_files/python/python-3.6.4.source
source /share/apps/source_files/cuda/cuda-9.0.source

export LD_LIBRARY_PATH="/share/apps/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/share/apps/python-3.6.4-shared/lib/:$LD_LIBRARY_PATH"
export PATH="$PATH:/share/apps/python-3.6.4-shared/bin/"
export PATH="/share/apps/cuda-9.0/bin:$PATH"
export VENV_FOLDER="/home/gdisciac/Python64Venv9"

cd /home/gdisciac/DOTPrior/
python3 -m venv $VENV_FOLDER
source $VENV_FOLDER/bin/activate
pip install --upgrade pip
pip install -r  requirements.txt

# frmw 1 slice is shared
#1
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw1b' 1 1 'BORNsvdLarge2' 1 'USadded'  ; 
#2
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice_withEXP_frmw1b' 0 1 'BORNsvdLarge2' 1 'USadded'  ;
#3
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic_withEXP_frmw1b' 1 0 'BORNsvdLarge2' 1 'USadded' ; 
#4
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw1b' 1 1 'FEMsvdLarge2' 1 'USadded' ; 
#5
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic_withEXP_frmw1b' 1 0 'FEMsvdLarge2' 1'USadded' ; 
#6
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw1b' 1 1 'FEMHETEsvdLarge2' 1 'USadded'; 
#7
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_optic_withEXP_frmw1b' 1 0 'FEMHETEsvdLarge2' 1 'USadded' ; 
# frmw3 optical is the same as frw1
#8
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw3b' 1 1 'BORNsvdLarge2' 1 'USonly' ; 
#9
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_slice_withEXP_frmw3b' 0 1 'BORNsvdLarge2' 1 'USonly';
#10
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw3b' 1 1 'FEMsvdLarge2' 1 'USonly'; 
#11
#python DeepPrior_OpticalLoss_iter.py 'OpticalLoss_iter1_full_withEXP_frmw3b' 1 1 'FEMHETEsvdLarge2'   1 'USonly'; 
# frmw4 slice is shared; optical is same as frmw1 
#12
python DeepPrior_OpticalLoss_iterUS.py 'OpticalLoss_iter1_full_withEXP_frmw4b' 1 1 'BORNsvdLarge2'  1 'USadded';
#13
#python DeepPrior_OpticalLoss_iterUS.py 'OpticalLoss_iter1_slice_withEXP_frmw4b' 0 1 'BORNsvdLarge2'  1  'USadded';
#14
#python DeepPrior_OpticalLoss_iterUS.py 'OpticalLoss_iter1_full_withEXP_frmw4b' 1 1 'FEMsvdLarge2'  1 'USadded';
#15
#python DeepPrior_OpticalLoss_iterUS.py 'OpticalLoss_iter1_full_withEXP_frmw4b' 1 1 'FEMHETEsvdLarge2' 1  'USadded';
echo end-script
