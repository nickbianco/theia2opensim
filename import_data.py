import os
import shutil
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

og_data_path = config['original_data_path']
og_jump_1_path = os.path.join(og_data_path, 'ACL Registry CMJ', 'Theia output data',
                              'Counter-Movement Jump Markerless 1')

data_path = config['data_path']
jump_1_path = os.path.join(data_path, 'acl', 'jump_1')
os.makedirs(jump_1_path, exist_ok=True)

# Copy all files from the original data path to the local data path.
for fname in os.listdir(og_jump_1_path):
    shutil.copy(os.path.join(og_jump_1_path, fname), jump_1_path)
