#!/bin/bash
#$ -l tmem=5G
#$ -l h_vmem=20G
#$ -l h_rt=2:00:00
#$ -S /bin/bash
#$ -j y
#$ -N mcxRefl1sam
#$ -l gpu=true
#$ -t 1:57600
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
export cmd="matlab -nodesktop -nodisplay -nosplash -nojvm -r Ilaunch="$SGE_TASK_ID";run('/home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/master_mcx_sh_refl_1sam.m');exit;"
#echo $cmd
echo Starting matlab
cd /home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/
$cmd 
#matlab -nodesktop -nodisplay -nosplash -nojvm -r Ilaunch="$SGE_TASK_ID";run('/home/gdisciac/mcx/mcxlab/examples/mcx_myscripts/master_mcx_sh_refl.m');exit;

echo end-script
