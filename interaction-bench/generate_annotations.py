import json
import argparse
import os
import random


def safe_sample(population, k):
    k = min(len(population), k)
    return random.sample(population, k)


def wrap_text(text, limit=80, prefix=""):
    wrapped_text = ""
    for i, word in enumerate(text.split(" ")):
        if len(wrapped_text.split("\n")[-1] + " " + word) > limit:
            wrapped_text += "\n" + prefix
        wrapped_text += word + " "

    return wrapped_text


all_pos_labels = "\n".join(
    [
        " - [" + label.upper() + "]"
        for label in ["acknowledgement", "followup", "overresponse", "display"]
    ]
)

all_neg_labels = "\n".join(
    [" - [" + label.upper() + "]" for label in ["clarification", "repair", "repeat"]]
)

all_uncategorized_labels = "\n".join(
    [" - [" + label.upper() + "]" for label in ["topic switch"]]
)

label_str = f"=> Delete SUCCESSFUL GROUNDING labels that DO NOT apply:\n{all_pos_labels}\n=> Delete UNSUCCESSFUL GROUNDING labels that DO NOT apply:\n{all_neg_labels}\n=> Delete UNCATEGORIZED labels that DO NOT apply:\n{all_uncategorized_labels}\n"


def create_text_conversation(conversation):
    conversation_text = f'{conversation["conversation_id"]}\n\n'
    conversation_text = "** TURN 0 **\n"
    for idx, message in enumerate(conversation["labels"]["messages"]):
        if message["role"] == "user":
            if idx == 0:
                conversation_text += (
                    "User: ".upper()
                    + wrap_text(message["message"], 80, "      ")
                    + "\n=> What is the users goal?: [ANSWER]\n=> Delete the labels that DO NOT apply:\n - [BEGIN]\n\n"
                )
            else:
                conversation_text += (
                    "User: ".upper()
                    + wrap_text(message["message"], 80, "      ")
                    + f"\n=> What is the users goal?: [ANSWER]\n{label_str}\n"
                )

        else:
            conversation_text += (
                message["role"].upper()
                + ": "
                + wrap_text(message["message"], 80, "      ")
                + f"\n{label_str}\n\n"
            )

        if idx % 2 != 0 and idx != len(conversation["labels"]["messages"]) - 1:
            conversation_text += f"** TURN {(idx // 2) + 1} **\n"

    return conversation_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="copilot")
    parser.add_argument(
        "--input_file", type=str, default="./data/raw/$dataset_sample_labeled_test.json"
    )
    parser.add_argument(
        "--output_folder", type=str, default="./data/to_annotate/$dataset"
    )
    args = parser.parse_args()

    input_file = args.input_file.replace("$dataset", args.dataset)
    output_folder = args.output_folder.replace("$dataset", args.dataset)

    with open(input_file) as f:
        data = json.load(f)

    high_repair = []
    low_repair = []

    for conversation in data:
        if conversation["labels"] is None:
            continue

        labels = conversation["labels"]["messages"]
        repair = 0
        for label in labels:
            if "labels" not in label:
                continue
            if any(x in label["labels"] for x in ["repair", "repeat", "clarification"]):
                repair += 1

        if repair > 2:
            high_repair.append(conversation)
        else:
            low_repair.append(conversation)

    # let's write these to the label_templated folder with each file name being the conversation_id.txt
    # sample 5 from each

    full_sample = safe_sample(high_repair, 5) + safe_sample(low_repair, 5)

    for idx, conversation in enumerate(full_sample):
        # create output folder if it doesn't exist

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(f"{output_folder}/{idx}.txt", "w") as f:
            f.write(create_text_conversation(conversation).strip())
