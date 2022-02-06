#######################################################
# Multi Processor SMP matlab example 8 Processors    #
# Note this example requests 16G [2x8] of memory    #
# Also note matlab is useless at memory management #
# If you use a parfor make sure you specify M=8   #
# and matlabpool open local 8		      	 #
# matlabpool will use 1.1G for each local worker#
# even before it has done any calculations     #
# Note matlabpool need jvm		      #
################################################

#$ -S /bin/bash
#$ -l h_vmem=2G
#$ -l tmem=2G
#$ -l h_rt=32:0:0
#$ -cwd
#$ -j y
#$ -R y
#$ -pe smp 8
#$ -N MyMatlabJob
date
hostname
#If you need lots of local i/o create a folder on scratch 0 and stage your data to it.
if [ ! -d "/scratch0/gdisciac" ]; then
	mkdir /scratch0/gdisciac
fi

/share/apps/matlabR2016b/bin/matlab -nodisplay -nodesktop  -nosplash < master.m 

