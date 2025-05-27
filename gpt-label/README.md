

# grounding

## downloading copilot files

you may need to pip install the following (in a fresh conda env), along with azure identity stuff.
```
pip install openai pandas pyarrow
```

if you want to do analysis on the copilot data, start by downloading files from the ext1 container. 
```
python -m src.download_blobs oshaikh/logs_parquet
```

TODO: dump conda requirements

## processing datasets into a unified format

Next, we're going to want to process all the datasets into a unified format. Every dataset (CoPilot, MultiWOZ, WildChat) has it's own creative way for encoding messages. The following script downloads the remaining datsets, reads everything and parses it into a unified format.

```
python -m src.extract_data
```

The format looks something like this

```
{
    "parsedMessages": [
        {
            "role": "user|assistant",
            "content": "message"
        }
    ],
    "ConversationId": [id1, id2, ...]
}
```

Each object is saved as a pickle file (.pkl) in /labeled_data

## annotating datasets

To annotate datasets, you'll need to run the following .sh file.

```
bash/label.sh
```

label.sh is very minimal in content.

```
python -m src.gpt_label \
    --batch_size 10 \
    --pickle "./labeled_data/multiwoz_sample.pkl" \
    --output "./labeled_data/multiwoz_sample_labeled.json" \
    --resume
```

It calls gpt_label.py, which labels each message with GPT 4o-mini. To speed this process up, you can batch requests asynchronously (using the batch_size query). The --resume flag just makes sure we don't overwrite our output. 

## analysis

All analysis of the labeled data is located in the notebooks directory (or will be very soon!). You can run them from top-to-bottom to generate plots.

## next steps.

Once we have our labeled JSON files, we can train a forecaster! See interaction-bench folder for more details. 

