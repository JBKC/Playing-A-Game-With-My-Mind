'''
Stream EEG data and run realtime inference on trained model
'''

from neurosity import NeurositySDK
import asyncio
import queue
from dotenv import load_dotenv
import numpy as np
import os
import crown_realtime_processing
import joblib
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

# task 1: pull EEG data
async def eeg_stream(data_queue, neurosity):
    # create callback that continuously extracts signals

    # Neurosity SDK expects a synchronous function
    def callback(data):
        signals = np.array(data['data'])
        # append data to queue
        data_queue.put(signals)
        print(f'QUEUE LENGTH: {data_queue.qsize()}')

    # initiate callback
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)

    # stream data indefinitely
    try:
        while True:
            # yield control back to event loop
            await asyncio.sleep(0)

    # stop the stream when program exits
    finally:
        unsubscribe()

# task 2: process data
async def eeg_processing(data_queue, model, W):
    window = []
    iter = 0
    iters = 100  # how many seconds of data to collect

    while True:
        try:
            # take the first datapacket in the queue
            signals = data_queue.get()
            iter += 1
            print(f'iter: {iter}')

            # add this data to a shifting window
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

        except asyncio.TimeoutError:
            print("Timeout waiting for data")
            continue

async def main():

    neurosity = await initialise()                    # initialise Crown headset
    data_queue = queue.Queue()                        # create FIFO queue to hold signal data

    # load saved model & spatial filters
    model_file = joblib.load('models/lda_2024-07-21 13:37:50.903718.joblib')
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




