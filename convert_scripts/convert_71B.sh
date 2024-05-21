#!/bin/bash

#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-01:00:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

SCRIPT_DIR="/scratch/project_462000319/rluukkon/megatron_tests/Megatron-LM-test2"

CONTAINER="/scratch/project_462000319/rluukkon/singularity/flash-attn-test-2_pems_v2.sif"
SING_BIND="/flash/project_462000319/,/scratch/project_462000319/,/scratch/project_462000086/,/scratch/project_462000353/,/scratch/project_462000444"

TOKENIZER="/scratch/project_462000444/europa/tokenizers/europa_tokenizer_262144_rc3-sampled-50B-shuf.jsonl"
CONFIG_FILE="/scratch/project_462000319/rluukkon/megatron_tests/Megatron-LM-test2/europa_71B.config"

mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

if [ -z "$1" ]; 
    then
    echo "You need to pass the checkpoint path as an argument!"
fi

# Give checkpoint path and output dir as positional args e.g. 
# "sbatch convert_scripts/convert_71B.sh /scratch/project_462000353/europa-checkpoints/1024N/iter_0007500/"

CHECKPOINT_PATH=$1
OUTPUT_ROOT="/scratch/project_462000086/risto/europa/converted_models"
#OUTPUT_ROOT="/scratch/project_462000086/risto/viking-v3/converted_models"
OUTPUT_DIR=$OUTPUT_ROOT/europa_71B_$(basename ${CHECKPOINT_PATH})_bfloat16
# OUTPUT_DIR=$


CMD="python3 $SCRIPT_DIR/tools/checkpoint/viking_conversion/convert_to_llama.py \
    --path_to_unmerged_checkpoint $CHECKPOINT_PATH \
    --config_file $CONFIG_FILE \
    --tokenizer $TOKENIZER \
    --output_dir $OUTPUT_DIR
    "

if [ -n "$SINGULARITY_ENVIRONMENT" ]; then
	srun \
		singularity exec \
		-B "$SING_BIND" \
		-B "$PWD" \
		"$CONTAINER" \
		bash -c "source /opt/miniconda3/bin/activate pytorch; \
		$CMD"
else
	singularity exec \
		-B "$SING_BIND" \
		-B "$PWD" \
		"$CONTAINER" \
		bash -c "source /opt/miniconda3/bin/activate pytorch; \
		$CMD"
fi
