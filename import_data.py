import os
import shutil
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

def import_data(original_trial_relpath, trial_relpath):
    original_data_path = config['original_data_path']
    data_path = config['data_path']

    original_trial_path = os.path.join(original_data_path, original_trial_relpath)
    trial_path = os.path.join(data_path, trial_relpath)
    os.makedirs(trial_path, exist_ok=True)

    # Copy all files from the original data path to the local data path.
    for fname in os.listdir(original_trial_path):
        shutil.copy(os.path.join(original_trial_path, fname), trial_path)
