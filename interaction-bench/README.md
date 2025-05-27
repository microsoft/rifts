
# InteractionBench

In this folder, we will train a grounding forecaster and extract probabilistic estimates of a prompt yielding some kind of grounding failure P(fix| PLEASE EXPLAIN). We'll also extract prompts that are representative of grounding failures and successes!


## generating training data for the forecaster.

To generate train data, you will need to copy the labeled data from the gpt-label folder. This is the data we labeled with GPT-4 and stored in the labeled_data folder. Specifically;

```
cp -r gpt-label/labeled_data/DATASET_sample_labeled.json interaction-bench/data/raw/
```

Next, you'll generate train/val/test splits for each dataset. We use a 33/33/33 split because we have a lot of data, and we'll want to train equally powerful train/val/test simulators. 

```
python generate_splits.py
```

Once you're done generating splits, you'll need to process the data and turn it into a format that torchtune---the training library we use---is happy with. Run the following next. The train data will be stored in ./data/processed/

```
python generate_train_data.py --dataset DATASET --split SPLIT
```

## training in the forecaster

Finally, you can start training simulators / forecasters on the data! 

First, you'll need to install a modified version of torchtune that lets us train on custom grounding tokens. The torchtune version lives in the torchtune folder. cd into the folder and run the following---this will make sure that you're install the modified version of torchtune from that specific folder. 

```
pip install --editable .
```

Next, use that version of torchtune to train models on our data. This will require 1 80GB GPU (e.g. an A100).
```
bash shell/start_train_model.sh DATASET SPLIT
```

## inference on the forecaster

To run inference on the forecaster, you'll need to spin up a vLLM server. To do that, run the following:

```
bash shell/start_inference.sh DATASET SPLIT CHECKPOINT
```

With this running in the background, we can now generate P(grounding act) for each prompt in our splits.

```
python generate_logits.py --checkpoint CHECKPOINT --dataset DATASET [--run_split val --train_split train]
```

This script will produce logits in ./data/logits/, which you can then analyze for an intervention!

## GPT-4 baselines

You can similarly produce zero-shot logits from GPT as a comparision point:

```
python generate_gpt_logits.py --dataset DATASET --split SPLIT
```

which will also add logits to ./data/logits/

## Analysis

All analysis of logits + forecaster ROC AUC is located in the notebooks directory (or will be very soon!). You can run them from top-to-bottom to generate plots.
