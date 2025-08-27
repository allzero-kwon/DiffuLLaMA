# export HF_TOKEN=<HF_TOKEN>
# export HF_HOME=<DIRECTORY TO YOUR HUGGINGFACE HOME>
# export HF_DATASETS_CACHE=<DIRECTORY TO YOUR CACHE FOLDER>

# export TRITON_LIBCUDA_PATH=<CUDA DIR> e.g. /usr/local/cuda/compat/lib.real 
# export CUDA_LAUNCH_BLOCKING=1

set -ex
export CUDA_DEVICE_MAX_CONNECTIONS=1
PBSNODEFILE=hostname.txt
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)

# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=5000
GPUS_PER_NODE=4
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #RANDOM

    # --machine_rank $NODEID \

config_json=accelerate_configs/single_node.yaml

export LAUNCHER="accelerate launch \
    --config_file $config_json \
    --main_process_ip $MASTER_ADDR \
    --num_processes 3 \
    --main_process_port $MASTER_PORT \
    --gpu_ids 1,2,3 \
    "

export CMD="train.py \
--batch-size 20 \
--gradient-accumulate-every 4  \
--output-dir ./output/1.5B_diffusion \
--seed 2829 \
--wandb Diffusion \
--max-train-steps 2000  \
--learning-rate 1.5e-5  \
--dataset /common_models1/da02/tinyllama_datasets \
--model /common_models1/da02/models/DeepSeek-R1-Distill-Qwen-1.5B  \
--seq-length 2048 \
--parallel_mode data_parallel \
"

export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
$LAUNCHER $CMD