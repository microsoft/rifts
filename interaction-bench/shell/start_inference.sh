dataset=$1
split=$2
checkpoint=$3

# do not run if both arguments are not provided

if [ -z "$split" ] || [ -z "$checkpoint" ]; then
    echo "Please provide split and checkpoint"
    exit 1
fi

mkdir -p ./checkpoints/inference_checkpoints/$split/$checkpoint

# copy tokenizer from checkpoints/simulator_checkpoints_${1}
cp ./checkpoints/simulator_checkpoints_${1}/$split/*.json ./checkpoints/inference_checkpoints/$split/$checkpoint

# Check if the file(s) already exist before copying
if [ -z "$(ls ./checkpoints/inference_checkpoints/$split/$checkpoint/hf_model_000*_${checkpoint}.pt 2>/dev/null)" ]; then
    echo "Copying!"
    cp -r ./checkpoints/simulator_checkpoints_${1}/$split/hf_model_000*_${checkpoint}.pt ./checkpoints/inference_checkpoints/$split/$checkpoint
else
    echo "Model checkpoint already exists, skipping copy."
fi

python -m entrypoints.api_server --model ./checkpoints/inference_checkpoints/$split/$checkpoint --dtype bfloat16  --port 1235 --max-logprobs 50
