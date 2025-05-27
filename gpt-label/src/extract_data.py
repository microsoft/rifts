from .helpers import load_dfs, filter_data
import pandas as pd
import json
from datasets import load_dataset
import random
import requests

#############################
# Copilot Log Extraction
#############################

# SAMPLE_SUBSET = 100

df = load_dfs()
df.to_pickle("./labeled_data/copilot_sample.pkl")

#############################
# MultiWoZ Extraction
#############################
base_url = (
    "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/"
)
data_files = {
    "./multiwoz/dev_dialogues_001.json": base_url + "dev/dialogues_001.json",
    "./multiwoz/dev_dialogues_002.json": base_url + "dev/dialogues_002.json",
    "./multiwoz/test_dialogues_001.json": base_url + "test/dialogues_001.json",
    "./multiwoz/test_dialogues_002.json": base_url + "test/dialogues_002.json",
}

# download each file
for local_file, url in data_files.items():
    with open(local_file, "wb") as f:
        f.write(requests.get(url).content)

final_map = {"ConversationId": [], "task": [], "parsedMessages": []}

for file, _ in data_files.items():
    with open(file) as f:
        data = json.load(f)

    for task in ["attraction", "bus", "hospital", "police"]:
        filtered_convos, filtered_ids = filter_data(data, task)

        final_map["ConversationId"].extend(filtered_ids)
        final_map["task"].extend([task] * len(filtered_ids))
        final_map["parsedMessages"].extend(filtered_convos)

df_multiwoz = pd.DataFrame(final_map)
df_multiwoz.to_pickle("./labeled_data/multiwoz_sample.pkl")

#############################
# WildChat Extraction
#############################

ds = load_dataset("allenai/WildChat-1M")
dataset = ds["train"]

english_only = []

for i, lang in enumerate(dataset["language"]):
    if lang == "English":
        english_only.append(i)

english_only_dataset = dataset.select(english_only)

ip_to_indices = {}

for i, ip in enumerate(english_only_dataset["hashed_ip"]):
    if ip not in ip_to_indices:
        ip_to_indices[ip] = []
    ip_to_indices[ip].append(i)

ip_to_indices = {ip: indices for ip, indices in ip_to_indices.items()}


sampled_indices = []

for ip, indices in ip_to_indices.items():
    sampled_indices.extend(random.sample(indices, 1))

sampled_dataset = english_only_dataset.select(sampled_indices)

all_convos = sampled_dataset["conversation"]
all_ids = sampled_dataset["conversation_hash"]

zipped_data = zip(all_convos, all_ids)


final_dict = {"parsedMessages": [], "ConversationId": []}
for convo, id in zipped_data:
    curr_convo = []
    for msg in convo:
        curr_msg = {
            "role": msg["role"],
            "content": msg["content"],
        }

        curr_convo.append(curr_msg)

    final_dict["parsedMessages"].append(curr_convo)
    final_dict["ConversationId"].append(id)


df_wildchat = pd.DataFrame(final_dict)
df_wildchat.to_pickle("./labeled_data/wildchat_sample.pkl")
