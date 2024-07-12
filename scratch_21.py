import json
import os

folder = 'test data'
for trial in os.listdir(folder):
    if 'eeg' in trial:
        with open(os.path.join('test data', trial), 'r') as f:
            data = json.load(f)

print(len(data))