'''
Stream EEG data and run realtime inference on trained model
'''

from neurosity import NeurositySDK
import asyncio
from dotenv import load_dotenv
import numpy as np
import os
import crown_realtime_processing
import joblib
import collections
import tracemalloc
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

def eeg_stream_callback(buffer):
    def callback(data):
        # retrieve new data
        signals = np.array(data['data'])
        # add data to sliding buffer
        for i in range(signals.shape[1]):
            buffer.append(signals[:, i].tolist())

        print(f'Buffer length: {len(buffer)}')

    return callback

# coroutine 1: pull EEG data
async def eeg_stream(buffer, neurosity):

    callback = eeg_stream_callback(buffer)
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)                 # neurosity expects synchronous callback

    try:
        while True:
            # yield control back to event loop after callback has been executed
            await asyncio.sleep(0)

    # stop stream when program exits
    finally:
        unsubscribe()

# coroutine 2: process EEG data
async def eeg_processing(buffer, maxlen, model, W):

    while True:
        # start processing if buffer is at maximum length
        if len(buffer) >= maxlen:
            # process data in buffer
            await asyncio.to_thread(crown_realtime_processing.main, buffer, model, W)

        await asyncio.sleep(0)

async def main():

    neurosity = await initialise()                    # initialise Crown headset
    maxlen = 32*16
    buffer = collections.deque(maxlen=maxlen)         # create sliding window (buffer)

    # load saved model & spatial filters
    model_file = joblib.load('models/lda_2024-07-21 13:37:50.903718.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # create tasks
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(eeg_stream(buffer, neurosity))
        task2 = tg.create_task(eeg_processing(buffer, maxlen, model, W))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())




