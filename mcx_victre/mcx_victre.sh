#!/bin/bash
#$ -l tmem=10G
#$ -l h_vmem=40G
#$ -l h_rt=50:00:00
#$ -S /bin/bash
#$ -j y
#$ -N mcx_bash.sh
#$ -l gpu=true
#$ -t 1:1#724

# qrsh -l tmem=10G,h_rt=00:10:00,gpu=true
#=========MATLAB===============
export PATH="/share/apps/matlabR2018b/bin:$PATH"
export PATH="/home/gdisciac/mcx/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export PATH="/share/apps/cuda-9.0/bin:$PATH"
export PATH="/share/apps/cuda-9.0:$PATH"
export PATH="/home/gdisciac/mcx/bin:$PATH"
#==========================
cd /home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/
echo $PATH

cmd="matlab -nodesktop -nodisplay -nosplash -nojvm -r run('/home/gdisciac/mcx_victre/launchMC_cluster($SGE_TASK_ID).m');exit;"
echo $cmd
echo Starting matlab
#cd /home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/
$cmd 

echo end-script
