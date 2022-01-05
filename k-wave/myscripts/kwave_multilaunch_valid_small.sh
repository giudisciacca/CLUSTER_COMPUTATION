#!/bin/bash
#$ -l tmem=5G
#$ -l h_vmem=60G
#$ -l h_rt=2:00:00
#$ -S /bin/bash
#$ -j y
#$ -N USvalarray.sh
#$ -l gpu=true,gpu_rtx2080ti=yes
#$ -l tscratch=10G
#$ -t 1-200
#=========MATLAB===============
export PATH="/share/apps/matlabR2016b/bin:$PATH"
export PATH="/home/gdisciac/mcx/bin:$PATH"
export EBROOTHDF5="/share/apps/hdf5-1.10.5/"
export CUDA_HOME="share/apps.cuda-10.0/"
export LD_LIBRARY_PATH="/share/apps/hdf5-1.10.5/lib:/share/apps/cuda-10.0/lib64:/share/apps/gcc-5.5/lib64:$LD_LIBRARY_PATH"
export PATH="/share/apps/hdf5-1.10.5/bin:/share/apps/gcc-5.5/bin:/share/apps/cuda-10.0/bin:$PATH"
export PATH="/share/apps/cuda-10.0:$PATH"
export PATH="/home/gdisciac/k-wave/binaries:$PATH"
chmod +x /home/gdisciac/k-wave/binaries/kspaceFirstOrder3D-CUDA

#==========================
cd /home/gdisciac/k-wave/myscripts/scripts_randomblob
echo $PATH
export ITER=$SGE_TASK_ID
export cmd="matlab -nodesktop -noFigureWindows -nodisplay -nosplash  -r  i0="$ITER";run('/home/gdisciac/k-wave/myscripts/scripts_randomblob/Cluster_run_smallExp_arrayvalid.m');exit;"
echo $cmd
echo Starting matlab
$cmd 

echo end-script
