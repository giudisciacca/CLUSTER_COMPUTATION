#!/bin/bash
#$ -l tmem=5G
#$ -l h_vmem=40G
#$ -l h_rt=48:00:00
#$ -S /bin/bash
#$ -j y
#$ -N python_launch.sh
#$ -l gpu=true

#=========MATLAB===============
source /share/apps/source_files/python/python-3.6.4.source
source /share/apps/source_files/cuda/cuda-9.0.source

export PATH="/home/gdisciac/.local/lib/python3.6/site-packages/:$PATH:/home/gdisciac/.local/lib/python3.6/site-packages/"
export PATH="/home/gdisciac/.local/bin:$PATH:/home/gdisciac/.local/bin"
export PATH="/share/apps/matlabR2018b/bin:$PATH"
export PATH="/home/gdisciac/mcx/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/apps/cuda-9.0/lib64"
export LD_LIBRARY_PATH="/share/apps/python-3.6.4-shared/lib/:/home/gdisciac/.local/lib/:$LD_LIBRARY_PATH:/share/apps/python-3.6.4-shared/lib/"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/apps/python-3.6.4-shared/lib/"
export PATH="/share/apps/cuda-9.0/bin:$PATH"
export PATH="$PATH:/share/apps/python-3.6.4-shared/bin/"
export PATH="/share/apps/cuda-9.0/bin:$PATH:/share/apps/cuda-9.0/bin"
#export PATH="$PATH:/share/apps/cuda-9.0"
#export PATH="/home/gdisciac/mcx/bin:$PATH"
export VENV_FOLDER = "/home/gdisciac/PythonVenv"

pip3 install -r  requirements.txt --user

cd /home/gdisciac/GaussBlur/
python3 -m venv $VENV_FOLDER
source $VENVFOLDER/bin/activate
 
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisox';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisox';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisox';
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisoy';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoy';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoy';
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisoz';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoz';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisox';
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisoxinv';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoxinv';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoxinv';
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisoyinv';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisoyinv';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisoyinv';
python3 DeepPrior_GaussBlur.py 'full' 1 1 's1anisozinv';
python3 DeepPrior_GaussBlur.py 'blur' 1 0 's1anisozinv';
python3 DeepPrior_GaussBlur.py 'slice' 0 1 's1anisozinv';

echo end-script
