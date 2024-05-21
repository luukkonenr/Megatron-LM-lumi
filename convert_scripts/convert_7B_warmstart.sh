#!/bin/bash

#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-00:30:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

CONTAINER="/scratch/project_462000319/rluukkon/singularity/flash-attn-test-2_pems_v2.sif"
SING_BIND="/flash/project_462000319/,/scratch/project_462000319/,/scratch/project_462000086/"
TOKENIZER=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072
CONFIG_FILE="/scratch/project_462000319/general-tools/viking_v2_conversion/config_files/viking_7b.json"
#CONFIG_FILE=./config_files/viking_7b.json

mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

if [ -z "$1" ]; 
    then
    echo "You need to pass the checkpoint path as an argument!"
fi

# Give checkpoint path and output dir as positional args
CHECKPOINT_PATH=$1
OUTPUT_ROOT="/scratch/project_462000086/risto/viking-v3/converted_models"
OUTPUT_DIR=$OUTPUT_ROOT/viking_v3_7B_warmstart_$(basename ${CHECKPOINT_PATH})_bfloat16
# OUTPUT_DIR=$


CMD="python3 tools/checkpoint/viking_conversion/_convert_to_llama.py \
    --path_to_unmerged_checkpoint $CHECKPOINT_PATH \
    --config_file $CONFIG_FILE \
    --tokenizer $TOKENIZER \
    --output_dir $OUTPUT_DIR
    "

srun \
    singularity exec \
    -B "$SING_BIND" \
    -B $PWD \
    "$CONTAINER" \
    bash -c "source /opt/miniconda3/bin/activate pytorch; \
    $CMD"
  
