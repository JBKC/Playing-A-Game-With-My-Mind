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
import time


# Load environment variables
load_dotenv()

async def initialise():
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

# task 1: pull EEG data
async def eeg_stream(data_queue, neurosity):
    # create callback that continuously extracts signals
    async def callback(data):
        signals = np.array(data['data'])
        # shift focus to eeg_processing
        await data_queue.put(signals)

    # initiate callback
    unsubscribe = await neurosity.brainwaves_raw_unfiltered(callback)

    # stream data indefinitely
    try:
        while True:
            pass
    # stop the stream when program exits
    finally:
        unsubscribe()


# task 2: process data
async def eeg_processing(data_queue, model, W):
    window = []
    iter = 0
    iters = 100  # how many seconds of data to collect

    while True:
        # take the first datapoint in the queue
        signals = await data_queue.get()
        iter += 1
        print(f'iter: {iter}')

        # add latest data to a shifting window
        for i in range(signals.shape[1]):
            window.append(signals[:, i].tolist())

        # if over 2 seconds of data collected, run inference on trailing 2 second window
        if iter > 32:
            # shift the window forward by one datapacket
            window = window[16:]
            # process data while allowing eeg_stream to collect more data
            await asyncio.to_thread(crown_realtime_processing.main, window, model, W)

        if iter >= 16 * iters:
            break

async def main():

    neurosity = await initialise()                      # initialise Crown headset
    data_queue = asyncio.Queue()                        # create queue for signal data to append to

    # load saved model & spatial filters
    model_file = joblib.load(
        'models/lda_2024-07-21 13:37:50.903718.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # create tasks
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(eeg_stream(data_queue, neurosity))
        task2 = tg.create_task(eeg_processing(data_queue, model, W))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())




