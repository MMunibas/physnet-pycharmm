#!/bin/bash    

###just for pc-studix0

#SBATCH --job-name=CPhOH-1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1
#SBATCH --mail-user kaisheng.song@unibas.ch ##########ADAPT
#SBATCH --mail-type=END,FAIL

module load intel/intel2018-openmpi3.1.1-cuda9.0

USER=`whoami`
#input file
input=run_tl_clphoh_699_avtz-1.inp ##############ADAPT

#restart folder (in case a run is restarted)
restart=

#datasets folder
datasets=datasets
#neural network code
neural_network=neural_network
#training handling code
training=training
#low-level models
models=models
#atomlabels file
atom_labels=atom_labels.tsv
#pythonscript for training
train=train.py
#environment
envi=/home/song/environments/physnet_gpu/bin/activate #GPU ############ADAPT eventually
#envi=env     #CPU

startfolder=`pwd`
scratch=/scratch/$USER.$SLURM_JOBID

#create scratch folder
if [ -d "$scratch" ]; then
   rm -rf $scratch
   echo "scratch directory exists already"
   mkdir $scratch
else
   echo "creating scratch directory"
   mkdir $scratch
fi

#copy existing restart folder if present
if [ -d "$restart" ]; then
   cp -r $restart $scratch
fi

#link/copy data to scratch folder and go there
cp -r $train $scratch
cp -r $input $scratch
cp -r $atom_labels $scratch
cp -r $models $scratch
cp -r $neural_network $scratch
cp -r $training $scratch
cp -r $datasets $scratch
#ln -s /home/song/CPhOH_JCP/NeuralNetwork/$datasets $scratch/$datasets  #######ADAPT
cd $scratch

#make necessary folders and load environment
source /home/song/environments/physnet_gpu/bin/activate
#conda activate physnet_gpu
#run actual jobs
export CUDA_VISIBLE_DEVICES=1
./$train @$input
#python3 $train @${input}
cp -r $scratch $startfolder

#remove scratch folder
#cd $startfolder
rm -r $scratch
