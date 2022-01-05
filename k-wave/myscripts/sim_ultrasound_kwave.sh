#!/bin/bash
#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=0:10:0
#$ -S /bin/bash
#$ -j y
#$ -N mcx_bash
#$ -l gpu=true

#=========MATLAB===============
export PATH="/share/apps/matlabR2018b/bin:$PATH"
#==============================

echo ${PATH}
cmd="matlab -nodisplay -nojvm -r run('/home/gdisciac/kwave/k-Wave/myscripts/master.m');exit;"
echo ${cmd}
${cmd} 

