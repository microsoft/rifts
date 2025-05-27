import os
import json
import math
import random
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from openai import OpenAI
from src.helpers import convert_convo_to_string
import anthropic

# =============================================================================
# Fallback for math.erfinv (for Python versions without math.erfinv)
# =============================================================================
try:
    erfinv = math.erfinv
except AttributeError:

    def erfinv(y):
        """
        Approximate inverse error function.
        Valid for -1 < y < 1.
        """
        if y <= -1 or y >= 1:
            raise ValueError("erfinv is only defined for -1 < y < 1")
        a = 0.147  # magic constant
        ln_term = math.log(1 - y * y)
        first_term = 2 / (math.pi * a) + ln_term / 2
        second_term = ln_term / a
        return math.copysign(
            math.sqrt(math.sqrt(first_term**2 - second_term) - first_term), y
        )


# =============================================================================
# Configuration Constants
# =============================================================================
DATASET_PATH = "./final_dataset.pkl"
GROUNDING_PROMPT_PATH = "./prompts/grounding_extended.txt"
OUTPUT_DIR = "./"

# Rate limiting parameters
MAX_CALLS_PER_SECOND = 10
ONE_SECOND = 1
MAX_WORKERS = 20

openai_client = OpenAI(
    api_key="YOUR_KEY_HERE",
)

# Initialize OpenAI client
together_client = OpenAI(
    base_url="https://api.together.xyz/v1/", api_key="YOUR_KEY_HERE"
)

# anthropic client
anthropic_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="YOUR_KEY_HERE",
)

# Load dataset and grounding prompt
df = pd.read_pickle(DATASET_PATH)
with open(GROUNDING_PROMPT_PATH, "r") as f:
    grounding_prompt = f.read()


# =============================================================================
# API Helper Functions
# =============================================================================
@sleep_and_retry
@limits(calls=MAX_CALLS_PER_SECOND, period=ONE_SECOND)
def call_api(*args, **kwargs):
    """
    Wrapper for the API call that enforces a maximum number of calls per second.
    """
    if "meta" in kwargs.get("model"):
        response = together_client.chat.completions.create(*args, **kwargs)
        return response.choices[0].message.content

    if "claude" in kwargs.get("model"):
        kwargs["max_tokens"] = 4096
        response = anthropic_client.messages.create(*args, **kwargs)
        return response.content[0].text

    response = openai_client.chat.completions.create(*args, **kwargs)
    return response.choices[0].message.content


def get_labels_from_convo(convo, prompt):
    """
    Given a conversation and a grounding prompt, calls the API to obtain labels.
    """
    split_prompt = prompt.split("PROMPT:")
    if len(split_prompt) < 2:
        raise ValueError("Grounding prompt is missing the 'PROMPT:' delimiter.")
    main_prompt = split_prompt[1].strip()

    sys_prompt_parts = split_prompt[0].split("SYSTEM:")
    if len(sys_prompt_parts) < 2:
        raise ValueError("Grounding prompt is missing the 'SYSTEM:' delimiter.")
    sys_prompt = sys_prompt_parts[1].strip()

    # Replace placeholder with the conversation string
    curr_main = main_prompt.replace(
        "{input_conversation}", convert_convo_to_string(convo)
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=4096,
        seed=0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": curr_main},
        ],
    )
    return response.choices[0].message.content


def process_row(row, model_name):
    """
    Process a single row: send the instruction to the model, get the conversation
    labels, and return the result.
    """
    prompt = row["instruction"]

    # First API call: get the model's response
    model_response = call_api(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    convo = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": model_response},
    ]

    # Second API call: get labels from the conversation
    label_str = get_labels_from_convo(convo, grounding_prompt)
    label_data = json.loads(label_str)

    return {
        "model": model_name,
        "model_response": model_response,
        "model_response_label": label_data["messages"][1]["labels"],
        "instruction": prompt,
        "instruction_label": row["label"],
    }


def benchmark_model_parallel(model_name):
    """
    Benchmarks the model by processing the test split in parallel.
    """
    results = {
        "model": [],
        "model_response": [],
        "model_response_label": [],
        "instruction": [],
        "instruction_label": [],
    }

    test_rows = df[df["split"] == "test"].iterrows()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_row, row, model_name): idx for idx, row in test_rows
        }

        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Processing rows",
        ):
            idx = future_to_index[future]
            try:
                res = future.result()
                results["model"].append(res["model"])
                results["model_response"].append(res["model_response"])
                results["model_response_label"].append(res["model_response_label"])
                results["instruction"].append(res["instruction"])
                results["instruction_label"].append(res["instruction_label"])
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results["model"].append(model_name)
                results["model_response"].append(None)
                results["model_response_label"].append(None)
                results["instruction"].append(df.iloc[idx]["instruction"])
                results["instruction_label"].append(df.iloc[idx]["label"])

    return results


def calculate_accuracy(benchmark_df):
    """
    Calculates overall accuracy based on the benchmark DataFrame.
    """
    correct = 0
    total = 0

    for _, row in benchmark_df.iterrows():
        total += 1
        instruction_label = row["instruction_label"]
        response_label = row["model_response_label"]

        if response_label is None:
            continue

        if instruction_label in ["addressing", "ambiguous"]:
            if "clarification" in response_label:
                correct += 1
        elif instruction_label == "advancing":
            if "followup" in response_label:
                correct += 1
        elif instruction_label == "none":
            if (
                "followup" not in response_label
                and "clarification" not in response_label
            ):
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def compute_confidence_interval(correct, total, confidence=0.95):
    """
    Compute the confidence interval for the accuracy using the normal approximation.
    """
    p = correct / total
    se = math.sqrt(p * (1 - p) / total)
    z = math.sqrt(2) * erfinv(confidence)
    lower_bound = p - z * se
    upper_bound = p + z * se
    return lower_bound, upper_bound


def calculate_per_label_accuracy(benchmark_df):
    """
    Calculates accuracy for each instruction label in the benchmark DataFrame.
    """
    label_stats = {}
    for _, row in benchmark_df.iterrows():
        instruction_label = row["instruction_label"]
        response_label = row["model_response_label"]
        if instruction_label is None:
            continue
        if instruction_label not in label_stats:
            label_stats[instruction_label] = {"correct": 0, "total": 0}
        label_stats[instruction_label]["total"] += 1

        if response_label is None:
            continue

        if instruction_label in ["addressing", "ambiguous"]:
            if "clarification" in response_label:
                label_stats[instruction_label]["correct"] += 1
        elif instruction_label == "advancing":
            if "followup" in response_label:
                label_stats[instruction_label]["correct"] += 1
        elif instruction_label == "none":
            if (
                "followup" not in response_label
                and "clarification" not in response_label
            ):
                label_stats[instruction_label]["correct"] += 1

    per_label_accuracy = {}
    for label, stats in label_stats.items():
        per_label_accuracy[label] = (
            stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        )
    return per_label_accuracy


def simulate_weighted_random_baseline(benchmark_df, weights=None):
    """
    Simulates the weighted random baseline on the test set.
    """
    if weights is None:
        weights = {"clarification": 1 / 3, "followup": 1 / 3, "none": 1 / 3}

    labels_list = list(weights.keys())
    weights_list = list(weights.values())

    baseline_correct = 0
    baseline_total = 0

    for _, row in benchmark_df.iterrows():
        true_label = row["instruction_label"]
        if true_label is None:
            continue
        baseline_total += 1

        predicted = random.choices(labels_list, weights=weights_list, k=1)[0]

        if true_label in ["addressing", "ambiguous"]:
            if predicted == "clarification":
                baseline_correct += 1
        elif true_label == "advancing":
            if predicted == "followup":
                baseline_correct += 1
        elif true_label == "none":
            if predicted == "none":
                baseline_correct += 1

    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0
    return baseline_accuracy, baseline_correct, baseline_total


def main(model_name):
    # Set the output file based on the provided model name.
    output_file = os.path.join(
        OUTPUT_DIR, f"benchmark_results_{model_name.replace('/', '_')}.json"
    )

    # Benchmark the model if results don't exist; otherwise, load the results.
    if not os.path.exists(output_file):
        print("Benchmarking model...")
        benchmark_results = benchmark_model_parallel(model_name)
        with open(output_file, "w") as f:
            json.dump(benchmark_results, f)
    else:
        print(f"Loading existing benchmark results from {output_file}...")
        with open(output_file, "r") as f:
            benchmark_results = json.load(f)

    # Convert results to DataFrame and calculate overall model accuracy.
    benchmark_df = pd.DataFrame.from_dict(benchmark_results)
    accuracy, correct, total = calculate_accuracy(benchmark_df)

    # Compute the 95% confidence interval for the model accuracy.
    lower, upper = compute_confidence_interval(correct, total, confidence=0.95)
    margin = (upper - lower) / 2

    print("Model Accuracy: {:.2%} ± {:.2%}".format(accuracy, margin))

    # Calculate and print per-label accuracy.
    per_label_accuracy = calculate_per_label_accuracy(benchmark_df)
    print("\nPer-label Accuracy:")
    for label, acc in per_label_accuracy.items():
        print("  {}: {:.2%}".format(label, acc))

    # Simulate the weighted random baseline using the default weights.
    baseline_accuracy, baseline_correct, baseline_total = (
        simulate_weighted_random_baseline(benchmark_df)
    )
    print("\nWeighted Random Baseline Accuracy: {:.2%}".format(baseline_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark a model using the provided model name."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to benchmark."
    )
    args = parser.parse_args()
    main(args.model_name)
