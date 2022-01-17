#!/bin/bash
#$ -l tmem=10G
#$ -l h_vmem=20G
#$ -l h_rt=36:00:00
#$ -S /bin/bash
#$ -j y
#$ -N FEMSIM.sh
#$ -t 1:12
#=========MATLAB===============
export PATH="/share/apps/matlabR2016b/bin/:share/apps/gcc-8.3/bin:$PATH"
#export PATH="/home/gdisciac/mcx/bin:$PATH"
#export EBROOTHDF5="/share/apps/hdf5-1.10.5/"
export CUDA_HOME="share/apps.cuda-9.0/"
#export LD_LIBRARY_PATH="/share/apps/hdf5-1.10.5/lib:/share/apps/cuda-10.0/lib64:/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/share/apps/toast-2.0.2/linux64/lib:$LD_LIBRARY_PATH
export PATH="/share/apps/hdf5-1.10.5/bin:/share/apps/gcc-5.5/bin:/share/apps/cuda-9.0/bin:$PATH"
export PATH="/usr/local/cuda-9.0/bin:$PATH"
export VICTRE="/home/gdisciac/VICTRE"
export TOASTDIR="/share/apps/toast-2.0.2"
#==========================


cd /home/gdisciac/FEM_simulations_diffnet/
echo $PATH
export cmd="matlab -nodesktop -nodisplay -nosplash -nojvm -r cd /home/gdisciac/SOLUS/;DOT_install;set_dir;cd /home/gdisciac/FEM_simulations_diffnet/;Ilaunch="$SGE_TASK_ID";run('/home/gdisciac/FEM_simulations_diffnet/FEM_generate4diffnet_new2.m');exit;"
echo Starting matlab
$cmd 


echo end-script
