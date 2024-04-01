#!/bin/bash
#YBATCH -r a100_1
#SBATCH -N 1
#SBATCH -J sampling
#SBATCH --time=168:00:00
#SBATCH --output /home/wangzq/workspace/job_log/%j.out
#SBATCH --error /home/wangzq/workspace/job_log/%j.err

. /etc/profile.d/modules.sh
module load openmpi/4.0.5 cuda/11.6 cudnn/cuda-11.6/8.4.1 nccl/cuda-11.6/2.11.4



#python train_diffusion.py --save_ckpts=True --do_val=False --wandb_on=True

python -u sample.py


# num of GPUs
#NGPUS=8
# num of processes per node
#NPERNODE=8

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"
#export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
#export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

#mpirun -npernode $NPERNODE -np $NGPUS \


