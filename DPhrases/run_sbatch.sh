#!/bin/bash -e
echo CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}
eval "$(conda shell.bash hook)"
conda activate dap
python "$@"
