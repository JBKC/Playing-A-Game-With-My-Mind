'''
Stream EEG data and run realtime inference on trained model
'''

import asyncio
from neurosity import NeurositySDK
from dotenv import load_dotenv
import numpy as np
import os
import crown_realtime_processing
import joblib
from datetime import datetime
import time
import random
import threading

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

# 16*16 data per second

def main():
    neurosity = initialise()
    stream = []

    global iter, complete, window
    iter = 0  # loop counter
    complete = False
    iters = 100                 # how many seconds of data to collect
    window = []

    # load saved model & spatial filters
    model_file = joblib.load(
        '/Users/jamborghini/Documents/PYTHON/neurosity_multicontrol/models/lda_2024-07-19 22:37:28.677998.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # pull & process EEG data
    def callback(data):
        global iter, complete, window
        iter += 1
        print(f'iter: {iter}')

        signals = np.array(data['data'])
        for i in range(signals.shape[1]):
            ## improve this code with dequeue (no for loop)
            window.append(signals[:, i].tolist())

        # if over 2 seconds of total data collected, create sliding window of previous 2 seconds
        if iter > 32:
            # create sliding window
            window = window[16:]
            # processing & model inference - separate thread?
            crown_realtime_processing.main(window, model, W)

        if iter >= 16 * iters:
            time.sleep(10)
            complete = True
            unsubscribe()

    # begin data callback
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)



if __name__ == '__main__':
    main()




