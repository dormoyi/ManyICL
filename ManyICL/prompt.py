import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI
import json

# set seed
random.seed(66)


def in_context_example(dataset_name, num_ex=3):
    prompt = ""
    image_paths = []
    num_img = []

    if dataset_name == 'xbd':
        train_path = "/scr/geovlm/xbd_train_canon_classification.json"
        with open(train_path) as f:
            data = json.load(f)

        # randomly sample num_ex examples
        data = random.sample(data, num_ex)

        for i in range(num_ex):
            example = data[i]
            inp = example["conversations"][0]['value']
            answer_str = example["conversations"][1]['value']
            metadata = example['metadata']
            image_paths += example['video']
            task = example['task']
            original_input_polygon = example['original_input_polygon']

            prompt += "EXAMPLE " + str(i+1) + "\n"
            prompt += inp + " Only answer with the category and do not explain your reasoning." + "\n"
            prompt += "ANSWER: " + answer_str + "\n\n"

            num_img.append(len(example['video']))

        return prompt, image_paths
    
    elif dataset_name == 'fmow':
        train_path="/scr/geovlm/fmow_high_res_train.json"
        with open(train_path) as f:
            data = json.load(f)

        # randomly sample num_ex examples
        data = random.sample(data, num_ex)

        for i in range(num_ex):
            example = data[i]
            inp = example["conversations"][0]['value']
            answer_str = example["conversations"][1]['value']
            metadata = example['metadata']
            image_paths += example['video']
            task = example['task']

            prompt += "EXAMPLE " + str(i+1) + "\n"
            prompt += inp + " Only answer with the category and do not explain your reasoning." + "\n"
            prompt += "ANSWER: " + answer_str + "\n\n"

            num_img.append(len(example['video']))

        return prompt, image_paths, num_img


    else:
        raise ValueError("Dataset not supported")


def work(
    model,
    SAVE_FOLDER,
    dataset_path,
    detail="auto",
    file_suffix="",
):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.

    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    num_shot_per_class[int]: number of demonstrating examples to include for each class, so the total number of demo examples equals num_shot_per_class*len(classes)
    location[str]: Vertex AI location e.g. "us-central1","us-west1", not used for GPT-series models
    num_qns_per_round[int]: number of queries to be batched in one API call
    test_df, demo_df [pandas dataframe]: dataframe for test cases and demo cases, see dataset/UCMerced/demo.csv as an example
    classes[list of str]: names of categories for classification, and this should match tbe columns of test_df and demo_df.
    class_desp[list of str]: category descriptions for classification, and these are the actual options sent to the model
    SAVE_FOLDER[str]: path for the images
    dataset_name[str]: name of the dataset used
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    file_suffix[str]: suffix for image filenames if not included in indexes of test_df and demo_df. e.g. ".png"
    """

    if 'xbd' in dataset_path:
        dataset_name = "xbd"
        num_images = 2
    elif 'fmow' in dataset_path:
        dataset_name = "fmow"
        num_images = 8
    else:
        raise ValueError("Dataset not supported")

    EXP_NAME = f"{dataset_name}_{model}_{dataset_path.split('/')[-1].split('.')[0]}"

    if model.startswith("gpt"):
        api = GPT4VAPI(model=model, detail=detail)
    else:
        assert model == "Gemini1.5"
        api = GeminiAPI(location="us-central1")

    with open(dataset_path) as f:
            data = json.load(f)

    results = {}

    for question in tqdm(data):
        qns_id = question["id"]
        inp = question["conversations"][0]['value']

        answer_str = question["conversations"][1]['value']
        metadata = question['metadata']
        image_paths = question['video']
        task = question['task']
        if dataset_name != 'fmow':
            original_input_polygon = question['original_input_polygon']

        system_prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n"""

        prompt = inp

        if dataset_name == 'xbd' or dataset_name == 'fmow':
            prompt += " Only answer with the category and do not explain your reasoning."


        in_context_prompt, in_context_image_paths, num_images = in_context_example(dataset_name)
        num_images.append(len(image_paths))

        for retry in range(3):
            if (
                (qns_id in results.keys())
                and (not results[qns_id]['predicted'].startswith("ERROR"))
            ):  # Skip if results exist and successful
                continue

            try:
                res = api(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    image_paths=image_paths,
                    in_context_prompt=in_context_prompt,
                    in_context_images=in_context_image_paths,
                    num_images_per_round=num_images,
                    real_call=True,
                    max_tokens=250,
                    
                )

                print("Prediciton: ", res)
                print("Ground truth: ", answer_str)

                if dataset_name != 'fmow':
                    results[qns_id] = {
                        "question": inp,
                        "predicted": res,
                        "ground_truth": answer_str,
                        "task": task,
                        "original_input_polygon": original_input_polygon
                    }
                else:
                    results[qns_id] = {
                        "question": inp,
                        "predicted": res,
                        "ground_truth": answer_str,
                        "task": task
                    }

            except Exception as e:
                res = f"ERROR!!!! {traceback.format_exc()}"
            except KeyboardInterrupt:
                previous_usage = results.get("token_usage", (0, 0, 0))
                total_usage = tuple(
                    a + b for a, b in zip(previous_usage, api.token_usage)
                )
                results["token_usage"] = total_usage
                with open(f"{EXP_NAME}.pkl", "wb") as f:
                    pickle.dump(results, f)
                exit()


    root_dir = '/deep/u/idormoy/eval/answers/'
    answer_path =  root_dir + EXP_NAME + ".json"   

    with open(answer_path, 'w') as f:
        json.dump(results, f, indent=4)

    
    print('\n')
    print(f"Results saved to {answer_path}")
    print("Size of results:", len(results))

# add in-context