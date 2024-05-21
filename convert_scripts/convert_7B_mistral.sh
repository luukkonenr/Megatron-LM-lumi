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
ROOT="/scratch/project_462000319/rluukkon/megatron_tests/Megatron-LM-test2/Mistral-7B-v0.2/models--mistral-community--Mistral-7B-v0.2/snapshots/2c3e624962b1a3f3fbf52e15969565caa7bc064a"
TOKENIZER=${ROOT}
CONFIG_FILE="/scratch/project_462000319/rluukkon/megatron_tests/Megatron-LM-test2/convert_scripts/configs/mistral_7B_config.json"


mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

if [ -z "$1" ]; 
    then
    echo "You need to pass the checkpoint path as an argument!"
fi

# Give checkpoint path and output dir as positional args
CHECKPOINT_PATH=$1
OUTPUT_ROOT="/scratch/project_462000086/risto/mistral_contd_pretraining/converted_checkpoints/roundtrip_conversion/"
mkdir -p $OUTPUT_ROOT
OUTPUT_DIR=$OUTPUT_ROOT/mistral_7B_eng_$(basename ${CHECKPOINT_PATH})_bfloat16
# OUTPUT_DIR=$


CMD="python3 tools/checkpoint/viking_conversion/convert_to_mistral.py \
    --path_to_unmerged_checkpoint $CHECKPOINT_PATH \
    --config_file $CONFIG_FILE \
    --tokenizer $TOKENIZER \
    --output_dir $OUTPUT_DIR
    "

# srun \
    singularity exec \
    -B "$SING_BIND" \
    -B $PWD \
    "$CONTAINER" \
    bash -c "source /opt/miniconda3/bin/activate pytorch; \
    $CMD"
  
