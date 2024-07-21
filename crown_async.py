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
        # pause signal extraction until coroutine completes
        await data_queue.put(signals)

    unsubscribe = await neurosity.brainwaves_raw_unfiltered(callback)
    try:
        while True:
            await asyncio.sleep(0.1)  # Adjust as needed
    finally:
        unsubscribe()

    # begin data callback

# task 2: process data
async def eeg_process(data_queue, model, W):
    window = []
    iter = 0
    iters = 100  # how many seconds of data to collect

    while True:
        signals = await data_queue.get()
        iter += 1
        print(f'iter: {iter}')

        for i in range(signals.shape[1]):
            window.append(signals[:, i].tolist())

        if iter > 32:
            window = window[16:]
            # Assuming crown_realtime_processing.main is CPU-intensive:
            await asyncio.to_thread(crown_realtime_processing.main, window, model, W)

        if iter >= 16 * iters:
            break

async def main():

    neurosity = await initialise()                      # initialise Crown headset
    data_queue = asyncio.Queue()                        # create queue for signal data to append to

    # load saved model & spatial filters
    model_file = joblib.load(
        '/Users/jamborghini/Documents/PYTHON/neurosity_multicontrol/models/lda_2024-07-21 13:37:50.903718.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # run coroutines
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(eeg_stream(data_queue, neurosity))
        task2 = tg.create_task(eeg_process(data_queue, model, W))

    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())




