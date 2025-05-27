from openai import OpenAI
from transformers import AutoTokenizer
import json
import copy
from tqdm import tqdm
import argparse
import os

# a bit of a lazy hack
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/inference_checkpoints/train/1")

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|finetune_right_pad_id|>": 128004,
    "<|step_id|>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eom_id|>": 128008,
    "<|eot_id|>": 128009,
    "<|python_tag|>": 128010,
    "<|image|>": 128011,
    "<|video|>": 128012,
}

NUM_RESERVED_SPECIAL_TOKENS = 256

RESERVED_TOKENS = {
    f"<|reserved_special_token_{2 + i}|>": 128013 + i
    for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS))
}

LLAMA3_SPECIAL_TOKENS = {**SPECIAL_TOKENS, **RESERVED_TOKENS}

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:1235/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

all_toks = [f"<|reserved_special_token_{(i)}|>" for i in range(0, 22)]
special_token_ids = [LLAMA3_SPECIAL_TOKENS[x] for x in all_toks]
to_hf_toks = {
    x: y for x, y in zip(all_toks, tokenizer.convert_ids_to_tokens(special_token_ids))
}

idx_to_label = {
    2: "addressing",
    3: "ambiguous",
    4: "advancing",
    5: "none",
}


def get_completion(
    prompt,
    model,
    max_tokens=512,
    temp=0.0,
    restricted=False,
    stop=["<|eot_id|>"],
):
    completion = client.completions.create(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        extra_body={
            "skip_special_tokens": False,
            "add_special_tokens": False,
            "allowed_token_ids": special_token_ids if restricted else None,
        },
        logprobs=50,
        temperature=temp,
        echo=False,
        logit_bias=None,  # None if restricted else {k: -1000 for k in special_token_ids},
        stop=stop,
    )

    return completion.choices[0]


def parse_special_tokens(str):
    # remove special token 0 and 1
    str = str.replace("<|reserved_special_token_0|>", "")
    str = str.replace("<|reserved_special_token_1|>", "")
    return idx_to_label[int(str[-3])]


def construct_classes(logprobs):
    decoded_str = ""
    curr_top = logprobs.top_logprobs[0]
    classes = {}
    max_prob = -9999
    max_hf_tok = None
    for i in range(2, 6):
        curr_tok = to_hf_toks[f"<|reserved_special_token_{i}|>"]
        curr_prob = curr_top[curr_tok]

        if curr_prob > max_prob:
            max_prob = curr_prob
            max_hf_tok = curr_tok

        classes[idx_to_label[i]] = curr_prob

    decoded_str += max_hf_tok
    decoded_str += "<|reserved_special_token_1|>"
    return decoded_str, classes


def convert_convo(msgs):
    toks = []
    for idx, msg in enumerate(msgs):
        # check if this is the last idx
        toks.extend(
            [
                "<|eot_id|>" if idx != 0 else "<|begin_of_text|>",
                "<|start_header_id|>",
                msg["role"],
                "<|end_header_id|>",
                msg["content"],
            ]
        )

    return toks


def rollout_convo(msgs, model):
    outputs = []
    for idx, msg in enumerate(msgs):
        if msg["role"] == "user":
            outputs.append({"role": "user", "content": "\n\n" + msg["content"]})
            toks = convert_convo(outputs)

            outputs[-1]["content"] = (
                outputs[-1]["content"] + "<|reserved_special_token_0|>"
            )

            toks = convert_convo(outputs)

            completion = get_completion(
                "".join(toks),
                model,
                max_tokens=11,
                restricted=True,
            )

            decoded_str, classes = construct_classes(completion.logprobs)
            outputs[-1]["content"] += decoded_str
            outputs[-1]["classes"] = classes

        else:
            outputs.append({"role": "assistant", "content": "\n\n"})
            toks = convert_convo(outputs)

            completion = get_completion("".join(toks), model)
            outputs[-1]["content"] += completion.text.strip()

    return outputs


def preprocess_convo(convo):
    # make a copy of the convo array
    new_convo = copy.deepcopy(convo)
    true_labels = None
    for idx, msg in enumerate(new_convo):
        if "content" in msg and "<|reserved_special_token_0|>" in msg["content"]:
            only_content, labels = msg["content"].split("<|reserved_special_token_0|>")
            msg["content"] = only_content
            true_labels = parse_special_tokens(labels)

    return new_convo, true_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run-split", type=str, default="val")
    parser.add_argument("--train-split", type=str, default="train")

    parser.add_argument(
        "--input-file",
        type=str,
        default="./data/processed/$dataset_simulator/$dataset_simulator_$run-split_subsample.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/logits/$dataset_sample_logits_$train-split_$checkpoint_$run-split.json",
    )
    args = parser.parse_args()

    # check if output file exists
    instr_to_classes = {}

    output_file = (
        args.output_file.replace("$run-split", args.run_split)
        .replace("$train-split", args.train_split)
        .replace("$checkpoint", str(args.checkpoint))
        .replace("$dataset", args.dataset)
    )
    input_file = args.input_file.replace("$run-split", args.run_split).replace(
        "$dataset", args.dataset
    )

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            instr_to_classes = json.load(f)

    with open(input_file, "r") as f:
        data = json.load(f)

    # cutoff_priors = compute_class_priors(data, message_cutoff=2)
    # offsets = compute_offsets(cutoff_priors)

    print(f"Total number of conversations: {len(data["messages"])}")

    to_generate = []
    full_convos = []
    for convo in data["messages"]:
        if convo[0]["content"] in instr_to_classes:
            continue

        to_generate.append(
            [
                {
                    "content": convo[0]["content"],
                    "role": convo[0]["role"],
                }
            ]
        )

        full_convos.append(convo)

    print(f"Number of conversations to generate: {len(to_generate)}")

    for idx, convo in enumerate(tqdm(to_generate)):
        curr_convo, true_labels = preprocess_convo(convo)
        rolled_out = rollout_convo(
            curr_convo,
            f"./checkpoints/inference_checkpoints/{args.train_split}/{args.checkpoint}",
        )

        curr_dict = {
            "logits": {},
            "true_labels": true_labels,
            "full_convo": full_convos[idx],
        }

        try:
            curr_gen = rolled_out[-1]["classes"]
        except:
            print("failed")
            continue

        curr_dict["logits"] = curr_gen

        instr_to_classes[curr_convo[0]["content"]] = curr_dict

        with open(output_file, "w") as f:
            json.dump(instr_to_classes, f, indent=4)
