dataset=$1
split=$2

tune download meta-llama/Meta-Llama-3.1-8B --output-dir /scr/biggest/hface_cache/Meta-Llama-3.1-8B --ignore-patterns "original/consolidated.00.pth"

mkdir -p ./checkpoints/simulator_checkpoints_${1}/${2}

cp /scr/biggest/hface_cache/Meta-Llama-3.1-8B/tokenizer* ./checkpoints/simulator_checkpoints_${1}/${2}/

tune run full_finetune_single_device --config ./8B_full_single_device.yaml \
checkpointer.output_dir=./checkpoints/simulator_checkpoints_${1}/${2}/ \
output_dir=./checkpoints/simulator_checkpoints_${1}/${2}/ \
dataset.data_files=./data/processed/${1}_simulator/${1}_simulator_${2}_subsample.json \

