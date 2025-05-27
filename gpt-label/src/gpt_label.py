import argparse
from src.helpers import get_labels_from_convo
import os
import json
import tqdm
import pandas as pd
import asyncio


async def batched_query(prompt, convos):
    tasks = [get_labels_from_convo(convo, prompt) for convo in convos]
    res = await asyncio.gather(*tasks)
    return res


def load_convo():
    return


async def main(args):
    with open(args.prompt, "r") as f:
        selected_prompt = f.read()

    labeled_data = []

    with open(args.pickle, "rb") as f:
        df_sampled = pd.read_pickle(f)

    if args.resume:
        if os.path.exists(args.output):
            with open(args.output, "r") as f:
                labeled_data = json.load(f)

            # filter out all the conversation_id that have already been labeled

            labeled_ids = set([x["conversation_id"] for x in labeled_data])

            print("Original DF length")
            print(len(df_sampled))

            df_sampled = df_sampled[~df_sampled["ConversationId"].isin(labeled_ids)]

            print("Filtered DF length")
            print(len(df_sampled))

    # use batched_query to query the model in batches of N
    for i in tqdm.tqdm(range(0, len(df_sampled), args.batch_size)):
        while True:
            try:
                curr_batch = df_sampled.iloc[i : i + args.batch_size]

                labels = await batched_query(
                    selected_prompt, curr_batch["parsedMessages"].to_list()
                )

                parsed_json_labels = []

                # re-add ConversationID to the labels, look out for JSON parsing errors
                for idx, label in enumerate(labels):
                    curr_obj = {
                        "conversation_id": curr_batch.iloc[idx]["ConversationId"],
                        "labels": None,
                    }

                    try:
                        curr_obj["labels"] = json.loads(label)
                    except Exception as e:
                        print(e)

                    parsed_json_labels.append(curr_obj)

                labeled_data.extend(parsed_json_labels)

                with open(f"{args.output}", "w") as f:
                    json.dump(labeled_data, f, indent=4)

                break

            except Exception as e:
                if "Error code: 400" in str(e):
                    print("Error code 400, skipping")
                    print(e)
                    break

                print(e)

                # let's wait a bit and sleep
                await asyncio.sleep(10)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--resume", action="store_true", help="Resume labeling from where you left off"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for querying the model"
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default="./labeled_data/copilot_sample.pkl",
        help="Path to the parquet file containing the sampled data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./labeled_data/copilot_labeled_data.json",
        help="Path to the output file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Path to the prompt file",
        default="./prompts/grounding_extended.txt",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
