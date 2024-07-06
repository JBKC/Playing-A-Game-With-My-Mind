'''
File create EEG training data for classification
'''

from neurosity import NeurositySDK
import os
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime
import numpy as np
import time

# Load environment variables
load_dotenv()

def initialise():
    '''
    Initialise Crown for receiving data
    :return: neurosity: instance of NeurositySDK
    '''

    try:
        # get environment variables

        # initialise NeurositySDK
        neurosity = NeurositySDK({
            "device_id": os.getenv("DEVICE_ID"),
            "timesync": True

        })

        # login to Neurosity account
        neurosity.login({
            "email": os.getenv("EMAIL"),
            "password": os.getenv("PASSWORD")
        })

        print("Successfully logged in")
        return neurosity

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

################################

iter = 0                        # loop counter
output = pd.DataFrame()     # all streamed data
data_stream = {}           # dictionary for each data packet


def main():

    trial = input("Trial type - 'raise right arm' or 'raise left arm': ")

    neurosity = initialise()

    def callback(data):

        global iter, data_stream, output

        trial_time = 3              # in seconds
        folder = f'data_{trial_time} seconds'
        os.makedirs(folder, exist_ok=True)

        # data_stream['time'] = data['info']['startTime']
        channels = data['info']['channelNames']
        signals = data['data']

        # reformat channel data
        for i, channel in enumerate(channels):
            data_stream[channel] = signals[i]

        # add data into output dataframe
        data_stream = pd.DataFrame(data_stream)
        data_stream = data_stream.drop(['F5', 'PO3', 'PO4', 'F6'], axis=1)
        output = pd.concat([output, data_stream], ignore_index=True)

        iter += 1
        print(f'Iteration: {iter}')
        print(output)

        if iter >= trial_time * 16:
            result = output.to_dict(orient='list')
            # Save to JSON
            with open(f'{folder}/{trial}_{trial_time} seconds_{datetime.now()}.json', 'w') as f:
                json.dump(result, f, indent=2)

            unsubscribe()

    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)

if __name__ == '__main__':
    main()

