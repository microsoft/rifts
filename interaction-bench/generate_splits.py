import json
import argparse
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./data/raw/")
    parser.add_argument("--output_folder", type=str, default="./data/raw/")
    args = parser.parse_args()

    # list files in input folder
    input_files = os.listdir(args.input_folder)

    # filter out files with "train", "val", "test" in the name
    input_files = [
        file
        for file in input_files
        if "train" not in file and "val" not in file and "test" not in file
    ]

    for input_file in input_files:
        with open(args.input_folder + input_file, "r") as f:
            data = json.load(f)

        # generate train / val / test splits
        train, test = train_test_split(data, test_size=0.33, random_state=42)
        train, val = train_test_split(train, test_size=0.5, random_state=42)

        with open(
            args.output_folder + input_file.replace(".json", "_train.json"), "w"
        ) as f:
            json.dump(train, f)

        with open(
            args.output_folder + input_file.replace(".json", "_val.json"), "w"
        ) as f:
            json.dump(val, f)

        with open(
            args.output_folder + input_file.replace(".json", "_test.json"), "w"
        ) as f:
            json.dump(test, f)

    print("Done!")
