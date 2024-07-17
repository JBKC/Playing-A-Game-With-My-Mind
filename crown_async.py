'''
Stream EEG data and run realtime inference on trained model
'''

import asyncio
from neurosity import NeurositySDK
from dotenv import load_dotenv
import json
import os
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

    global iter, complete
    iter = 0  # loop counter
    complete = False
    iters = 16*10

    # pull EEG data
    def callback(data):
        global iter, complete
        stream.append(data)
        iter += 1
        print(f'iter: {iter}')
        if iter >= iters:
            complete = True
            unsubscribe()

    # begin data callback
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)



if __name__ == '__main__':
    main()




