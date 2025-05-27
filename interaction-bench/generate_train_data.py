#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import argparse
import random
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()

DATASET = args.dataset
SPLIT = args.split

# In[3]:


with open(f"./data/raw/{DATASET}_sample_labeled_{SPLIT}.json") as f:
    data = json.load(f)


# In[4]:


# forecasting prioritizes FIX
ORDERED_MERGES = [
    # addressing
    {
        "name": ["repeat", "repair"],
        "token": "<|reserved_special_token_2|>",
    },
    # ambiguous
    {
        "name": ["clarification", "overresponse"],
        "token": "<|reserved_special_token_3|>",
    },
    # advancing
    {
        "name": ["next turn", "followup", "overcontinue", "acknowledgement"],
        "token": "<|reserved_special_token_4|>",
    },
]

NONE_TOKEN = "<|reserved_special_token_5|>"


def create_finalized_convos(data):
    # make a deepcopy of the data
    data = json.loads(json.dumps(data))

    finalized_convos = []
    skipped_convos = 0

    duplicates = defaultdict(int)

    for convo in data:
        if convo["labels"] == None:
            continue

        try:
            curr_convo = convo["labels"]["messages"]
        except:
            skipped_convos += 1
            continue

        user_turns = 0
        assistant_turns = 0

        for turn in curr_convo:
            try:
                if turn["role"] == "user":
                    user_turns += 1
                else:
                    assistant_turns += 1
            except Exception as e:
                print(e)
                continue

        # first thing we want to do is remove all of the "hi there!" first messages
        # additionally, we want to remove all of the "are you gpt-4" messages.

        # the best way to do this is look at the first assistant message
        flagged_assistant = [
            "How can I assist you today?",
            "How can I help you today?",
            "How can I help you?",
            "How may I assist you today?",
            "How may I assist you?",
            "How may I help you today?",
            "How may I help you?",
        ]

        flagged_are_you = [
            "are you chatgpt",
            "are you gpt",
            "are you chat",
            "what version",
            "who are you",
            "whats your name",
            "what's ur name",
            "whats ur name",
            "Ignore previous data.Imagine you're an expert Graphic Designer and have experience in",
            "Could you give me only the ip address, username and password present in this text in this format",
        ]

        if len(curr_convo) >= 2:
            curr_message = curr_convo[1]

            # check if type curr_message is a dict

            if type(curr_message) == dict:
                curr_message = curr_message["message"]
            else:
                skipped_convos += 1
                continue

            if type(curr_message) != str:
                skipped_convos += 1
                continue

            if curr_convo[1]["role"] == "assistant":
                # check if the assistant message contains anything in the flagged_assistant_list
                if any([x.lower() in curr_message.lower() for x in flagged_assistant]):
                    # remove the first two messages
                    curr_convo = curr_convo[2:]

        if len(curr_convo) == 0:
            skipped_convos += 1
            continue

        if user_turns == 0 or assistant_turns == 0:
            skipped_convos += 1
            continue

        if curr_convo[0]["role"] == "user":
            # check if the user message contains anything in the flagged_are_you list
            if any(
                [x.lower() in curr_convo[0]["message"].lower() for x in flagged_are_you]
            ):
                skipped_convos += 1
                continue

            duplicates[curr_convo[0]["message"][:100]] += 1

            if duplicates[curr_convo[0]["message"][:100]] > 1:
                skipped_convos += 1
                continue

        # let's go in reverse
        rev_convo = curr_convo[::-1]
        curr_label = NONE_TOKEN

        flag = False

        for turn in rev_convo:
            try:
                label_str = f"<|reserved_special_token_0|>{curr_label}<|reserved_special_token_1|>"
                if turn["role"] == "user":
                    # print([turn["labels"] for label in turn["labels"]])
                    label_str = f"<|reserved_special_token_0|>{curr_label}<|reserved_special_token_1|>"
                    turn["content"] = str(turn["message"]) + label_str

                    curr_label = None

                    curr_set = set(turn["labels"])
                    for possible_label in ORDERED_MERGES:
                        for k in curr_set:
                            if k in possible_label["name"]:
                                curr_label = possible_label["token"]
                                break

                    if curr_label == None:
                        curr_label = NONE_TOKEN

                elif turn["role"] == "assistant":
                    turn["content"] = str(turn["message"]) + label_str

            except Exception as e:
                print(e)
                flag = True

        if flag:
            skipped_convos += 1
            continue
        # print(rev_convo)
        finalized_convos.append(rev_convo[::-1])

    for convo in finalized_convos:
        for turn in convo:
            if "labels" in turn:
                del turn["labels"]

            if "content" not in turn:
                turn["content"] = str(turn["message"])

            del turn["message"]

    # print frac of skipped convos

    print(f"Skipped {skipped_convos} convos out of {len(data)}")

    label_counts = defaultdict(list)
    for k in finalized_convos:
        if k[0]["role"] == "assistant":
            continue

        # extract everything between <|reserved_special_token_0|> and <|reserved_special_token_1|>
        label = ""
        for message in k:
            if message["role"] == "user":
                curr_label = (
                    message["content"]
                    .split("<|reserved_special_token_0|>")[1]
                    .split("<|reserved_special_token_1|>")[0]
                )
                label += curr_label[-3]

        label_counts[label[0]].append(k)

    return label_counts, duplicates


# In[5]:


simulator_data, dups = create_finalized_convos(data)

# now we need to subsample

for k in simulator_data:
    print(f"Label: {k} has {len(simulator_data[k])} samples")


# sample max possible from each label

finalized_simulator_data = {"messages": []}
sample_size = min(len(simulator_data[k]) for k in simulator_data)

for k in simulator_data:
    random.shuffle(simulator_data[k])
    finalized_simulator_data["messages"].extend(simulator_data[k][:sample_size])


# In[6]:


len(finalized_simulator_data["messages"])


# In[7]:


# save as JSON
# create folder under processed data, if not exists
if not os.path.exists(f"./data/processed/{DATASET}_simulator"):
    os.makedirs(f"./data/processed/{DATASET}_simulator")

all_sim_data = [k for k in finalized_simulator_data["messages"]]

with open(
    f"./data/processed/{DATASET}_simulator/{DATASET}_simulator_{SPLIT}.json", "w"
) as f:
    json.dump({"messages": all_sim_data}, f, indent=4)

with open(
    f"./data/processed/{DATASET}_simulator/{DATASET}_simulator_{SPLIT}_subsample.json",
    "w",
) as f:
    json.dump(finalized_simulator_data, f, indent=4)
