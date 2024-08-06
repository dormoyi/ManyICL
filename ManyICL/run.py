import pandas as pd
import argparse
from prompt import work


# python3 ManyICL/run.py --model="gpt-4o-2024-05-13" --dataset="/scr/geovlm/xbd_test_canon_classification.json"

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Experiment script.")
    # Adding the arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="UCMerced",
        help="The dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="Gemini1.5",
        help="The model to use",
    )


    # Parsing the arguments
    args = parser.parse_args()

    # Using the arguments
    dataset_name = args.dataset
    model = args.model

    # Folder to load the images, and this will be prepended to the filename stored in the index column of the dataframe.
    IMAGE_FOLDER = f"ManyICL/dataset/{dataset_name}/images"

    work(
        model,
        IMAGE_FOLDER,
        dataset_name
    )
