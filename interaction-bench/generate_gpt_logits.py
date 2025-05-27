# we're gonna take the first instruction from these datasets and produce similar confidences.
import json
import pdb
import asyncio
import argparse
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tqdm import tqdm

default_credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
token_provider = get_bearer_token_provider(
    default_credential, "https://cognitiveservices.azure.com/.default"
)

client = AsyncAzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="",
    azure_deployment="gpt4o-1",
    azure_ad_token_provider=token_provider,
)

token_remap = {2: "FIX", 3: "FOLLOWUP", 4: "CONTINUE", 5: "LEAVE"}


async def get_confidence_for_message(client, input_prompt):
    response = await client.chat.completions.create(
        # model="gpt-4o-2024-05-13", # model = "deployment_name".
        model="gpt-4o-2024-05-13",
        # model="gpt-4-0125",
        temperature=0.0,
        max_tokens=1,  #
        logprobs=True,
        top_logprobs=5,
        logit_bias={"50256": -100},
        seed=0,
        messages=[{"role": "user", "content": input_prompt}],
    )

    return response


async def process_confidences(confidences, dataset, split, prompt, client, messages):
    BATCH_SIZE = 5
    for idx in tqdm(range(0, len(messages), BATCH_SIZE)):
        batch = messages[idx : idx + BATCH_SIZE]
        # Skip if we've already processed this message
        prompts = []

        for message in batch:
            if message["content"] in confidences:
                print("skipping")
                continue

            curr_prompt = prompt.replace("{instruction}", message["content"])
            prompts.append(curr_prompt)

        # If we have no prompts to process, skip
        if len(prompts) == 0:
            continue

        while True:
            try:
                # Run the batch asynchronously
                tasks = [
                    get_confidence_for_message(client, prompt) for prompt in prompts
                ]
                results = await asyncio.gather(*tasks)
                break
            except Exception as e:
                print(e)
                print("Retrying")
                await asyncio.sleep(10)

        try:
            processed_responses = [
                result.choices[0].logprobs.content[0].top_logprobs for result in results
            ]
            idx = 0
        except:
            pdb.set_trace()

        for message, response in zip(batch, processed_responses):
            token_logprob_pair = {x.token: x.logprob for x in response}
            confidences[message["content"]] = {
                "label": message["label"],
                "prompts": prompts[idx],
                "confidences": token_logprob_pair,
            }
            idx += 1

        # write confidences to a file
        with open(f"./data/confidences/{dataset}_{split}.json", "w") as outfile:
            json.dump(confidences, outfile, indent=4)

    return confidences


def process_messages(data):
    # Get the first message in each conversation that's a user message
    first_messages = []
    prefix_len = len("<|reserved_special_token_")
    for conversation in data["messages"]:
        for message in conversation:
            if message["role"] == "user":
                msg, label = message["content"].split("<|reserved_special_token_0|>")
                first_messages.append(
                    {
                        "content": msg,
                        "label": token_remap[int(label[prefix_len:][0])].lower(),
                    }
                )
                break

    return first_messages


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wildchat")
    parser.add_argument("--split", type=str, default="val")

    args = parser.parse_args()

    dataset = args.dataset
    split = args.split

    with open(
        f"./data/processed/{dataset}_simulator/{dataset}_simulator_{split}_subsample.json"
    ) as f:
        eval_data = json.load(f)

    with open(
        f"./data/processed/{dataset}_simulator/{dataset}_simulator_train_subsample.json"
    ) as f:
        train_data = json.load(f)

    first_train_messages = process_messages(train_data)
    first_messages = process_messages(eval_data)

    # sample a few examples for the prompt from each category

    few_shot_examples = []

    for label in token_remap.values():
        for message in first_train_messages:
            if message["label"] == label.lower():
                few_shot_examples.append(
                    {"label": label, "content": message["content"]}
                )
                break

    # format the few-shot examples with content: [CONTENT] and label: [LABEL]
    few_shot_examples = "\n\n".join(
        [
            f"content: {example['content']}\nlabel: {example['label']}"
            for example in few_shot_examples
        ]
    )

    # load ./generate_confidence_prompt.txtpdb
    with open("./prompts/generate_confidence_prompt.txt", "r") as f:
        prompt = f.read()

    # replace the few-shot examples in the prompt
    prompt = prompt.replace("{few_shot_examples}", few_shot_examples)

    # open confidences.json if exists

    try:
        with open(f"./data/confidences/{dataset}_{split}.json") as f:
            confidences = json.load(f)
    except FileNotFoundError:
        confidences = {}

    # Now we process the messages in batches asynchronously
    confidences = await process_confidences(
        confidences, dataset, split, prompt, client, first_messages
    )


if __name__ == "__main__":
    asyncio.run(main())
