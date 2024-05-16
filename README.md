# ManyICL: Many-Shot In-Context Learning in Multimodal Foundation Models

<p>
    <a href='https://arxiv.org/abs/2404.09797' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</p>

This repository contains implementation of [ManyICL](https://arxiv.org/abs/XXX). Prepare a dataframe, configure your API key, modify the prompt and just run it!

Please note that this code repo is intended for research purpose, and might not be suitable for large-scale production.


# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```

# Setup API keys
## For GPT-series models offered by OpenAI
1. Get your API key from [here](https://platform.openai.com/api-keys);
2. Replace the placeholder in `ManyICL/LMM.py` (Line 36);

## For Gemini-series models offered by Vertex AI
Note that you need a Google cloud project for this. 
1. In the Google Cloud console, go to the [Dashboard](https://console.cloud.google.com/home).
2. Click the project selection list at the top of the page. In the Select a resource window that appears, select a project. Note the project ID displayed in the Project info section.
3. Replace the placeholder in `ManyICL/LMM.py` (Line 121);
4. If you're developing locally or on Colab (not on GCP instances), you need to authenticate by following this [instruction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/sdk-for-gemini/gemini-sdk-overview-reference#authenticate-vertex-python-sdk).

# Dataset preparation
Prepare two pandas dataframe: one for the demonstration set and one for the test set. You can find examples under the `dataset/` folder. Note that the index column should contain the filenames of the images. Here's a quick preview: 

| Index | Forest | Golf course | Freeway |
|:-------------|:--------------:|:--------------:|:--------------:|
|forest39.jpeg| 1 | 0 | 0 |
|golfcourse53.jpeg| 0 | 1 | 0 |
|freeway97.jpeg| 0 | 0 | 1 |

## Expected directory structure
Note that we only include 42 images in UCMerced dataset for illustration purposes. 

```
ManyICL/
├── LMM.py
├── dataset
│   └── UCMerced
│       ├── demo.csv
│       ├── test.csv
│       ├── images
│       │   ├── forest39.jpeg
│       │   ├── forest47.jpeg
│       │   ├── freeway09.jpeg
│       │   ├── freeway97.jpeg
│       │   ├── ...
├── prompt.py
└── run.py

```

# Configure the prompt

Modify the prompt in `prompt.py` if needed.

# Run the experiment
Run the experiment script, and it'll save all the raw responses in `UCMerced_21shot_Gemini1.5_1.pkl`.
```bash
python3 ManyICL/run.py --dataset=UCMerced --num_shot_per_class=1 --num_qns_per_round=3
```

# Citation

If you find our work useful in your research please consider citing:

```
TBD
```

## Acknowlegements
We thank Dr. Jeff Dean, Yuhui Zhang, Dr. Mutallip Anwar, Kefan Dong, Rishi Bommasani, Ravi B. Sojitra, Chen Shani and Annie Chen for their feedback on the ideas and manuscript. 
