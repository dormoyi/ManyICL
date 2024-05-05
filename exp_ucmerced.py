import pandas as pd
import argparse
from prompt import work

IMAGE_FOLDER = 'dataset/UCMerced_21'
demo_df = pd.read_pickle('dataset/UCMerced_demo_21.pkl')
test_df = pd.read_pickle('dataset/UCMerced_test_21.pkl')
classes = list(demo_df.columns)
class_desp = classes
class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
file_suffix = ''

exclude = {'GPT':[], 'Gemini':[] }
dataset_name = 'UCMerced'



if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Experiment script.')
    # Adding the arguments
    parser.add_argument('--model', type=str, required=False, default='Gemini1.5',
                        help='The model to use')
    parser.add_argument('--location', type=str, required=False, default='us-central1',
                        help='The location for the experiment')
    parser.add_argument('--num_shot_per_class', type=int, required=True,
                        help='The number of shots per class')
    parser.add_argument('--num_qns_per_round', type=int, required=False, default=1,
                        help='The number of questions asked each time')
    
    # Parsing the arguments
    args = parser.parse_args()

    # Using the arguments
    model = args.model
    location = args.location
    num_shot_per_class = args.num_shot_per_class
    num_qns_per_round = args.num_qns_per_round
    
    work(model, num_shot_per_class, location, num_qns_per_round, test_df, demo_df, classes, class_desp, IMAGE_FOLDER, file_suffix, exclude, dataset_name)
