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
SING_BIND="/flash/project_462000319/,/scratch/project_462000319/,/scratch/project_462000086/,/scratch/project_462000353/"

TOKENIZER=/flash/project_462000319/europa-tokenizer

mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

# Give checkpoint path and output dir as positional args
CHECKPOINT_PATH=/scratch/project_462000353/europa-checkpoints/1024N/iter_0007500/
OUTPUT_ROOT="/scratch/project_462000086/risto/europa/restructured_models"

OUTPUT_DIR=${OUTPUT_ROOT}/europa_71B_non_interleaved_pp4/$(basename ${CHECKPOINT_PATH})



CMD="python3 $SCRIPT_DIR/tools/checkpoint/restructure_interleaved_ckpt.py \
    --checkpoint_path $CHECKPOINT_PATH \
	--new_pp_size 4 \
    --output_path $OUTPUT_DIR \
    "

srun \
	singularity exec \
	-B "$SING_BIND" \
	-B "$PWD" \
	"$CONTAINER" \
	bash -c "source /opt/miniconda3/bin/activate pytorch; \
	$CMD"
