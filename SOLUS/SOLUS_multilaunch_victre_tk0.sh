#!/bin/bash
#$ -l tmem=400M
#$ -l h_vmem=20G
#$ -l h_rt=5:00:00
#$ -S /bin/bash
#$ -j y
#$ -N TK0_victre.sh
#$ -l gpu=true#,gpu_rtx2080ti=yes
##$ -l tscratch=20G
#$ -t 1:730#-8#730#95-99 #282#255,277#246,253#,255,277,282

#### qrsh -l tmem=14G,gpu=true,h_rt=0:30:0

#=========MATLAB===============
export PATH="/share/apps/matlabR2016b/bin/:share/apps/gcc-8.3/bin:$PATH"
#export PATH="/home/gdisciac/mcx/bin:$PATH"
#export EBROOTHDF5="/share/apps/hdf5-1.10.5/"
export CUDA_HOME="share/apps.cuda-9.0/"
#export LD_LIBRARY_PATH="/share/apps/hdf5-1.10.5/lib:/share/apps/cuda-10.0/lib64:/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/share/apps/toast-2.0.2/linux64/lib:$LD_LIBRARY_PATH
export PATH="/share/apps/hdf5-1.10.5/bin:/share/apps/gcc-5.5/bin:/share/apps/cuda-9.0/bin:$PATH"
export PATH="/usr/local/cuda-9.0/bin:$PATH"
export PATH="/home/gdisciac/k-wave/binaries:$PATH"
export VICTRE="/home/gdisciac/VICTRE"
export TOASTDIR="/share/apps/toast-2.0.2"
#==========================
cd /home/gdisciac/SOLUS
source $TOASTDIR/toastenv.sh
echo $PATH
export ITER=$SGE_TASK_ID
#export cmd="matlab -nodesktop -noFigureWindows -nodisplay -nosplash -r gpuDevice([]);i0="$ITER";set_dir;cd('/home/gdisciac/SOLUS/example/VICTRE_PARADIGM');multiple_run_paradigm_cluster;exit;"
echo $cmd
echo Starting matlab

matlab -nodesktop -noFigureWindows -nodisplay -nosplash -r "gpuDevice([]);i0=$ITER;DOT_install;set_dir;cd('/home/gdisciac/SOLUS/example/VICTRE_PARADIGM');tk0multiple_run_paradigm_cluster;exit;"

echo end-script
