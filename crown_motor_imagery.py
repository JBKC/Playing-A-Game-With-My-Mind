'''
File to map EEG kinetic thought training to button outputs
'''

from neurosity import NeurositySDK
import os
from dotenv import load_dotenv
import pandas as pd
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
# band 1 = 8-12Hz, band 2 = 13-24Hz, band 3 = 25-30Hz
# therefore 3 bands * 8 electrodes = 24 feature vectors in total
band_idx = {
    'band_1': [4, 5, 6],
    'band_2': [7, 8, 9, 10, 11, 12],
    'band_3': [13, 14, 15],
}

def callback(data):

    global iter, band_idx

    # get PSD feature vectors
    for electrode in data['psd']:
        psd_1 = [electrode[i] for i in band_idx['band_1']]
        psd_2 = [electrode[i] for i in band_idx['band_2']]
        psd_3 = [electrode[i] for i in band_idx['band_3']]

    iter += 1
    print(f"Iteration: {iter}")



def main():
    neurosity = initialise()
    neurosity.brainwaves_psd(callback)

if __name__ == '__main__':
    main()

