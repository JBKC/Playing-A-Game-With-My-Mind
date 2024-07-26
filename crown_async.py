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

async def put_buffer(buffer, signals):
    await buffer.put(signals)

def eeg_stream_callback(buffer, loop):
    def callback(data):
        # retrieve new data
        signals = np.array(data['data'])
        # append data to queue on the event loop
        asyncio.run_coroutine_threadsafe(put_buffer(buffer, signals), loop)
        print(f'Buffer size: {buffer.qsize()}')

    return callback

# coroutine 1: pull EEG data (producer)
async def eeg_stream(buffer, neurosity):

    loop = asyncio.get_running_loop()  # get event loop
    callback = eeg_stream_callback(buffer, loop)
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)         # neurosity expects synchronous callback

    try:
        while True:
            # yield control back to event loop after callback has been executed
            await asyncio.sleep(0)

    # stop stream when program exits
    finally:
        unsubscribe()
        await buffer.put(None)          # signal end of stream

# coroutine 2: process EEG data (consumer)
async def eeg_processing(buffer, maxlen, model, W):

    queue = collections.deque(maxlen=maxlen)         # create sliding window (double ended queue)

    while True:
        # get latest entry in buffer
        item = await buffer.get()
        print(f'Window (queue) size: {len(queue)}')
        # if final iteration, terminate coroutine
        if item is None:
            break

        # append latest item to window
        queue.append(item)

        # start processing at the point where queue reaches max length
        if len(queue) == 32:
            batch = np.concatenate(queue, axis = 1)
            await asyncio.to_thread(crown_realtime_processing.main, batch, model, W)

async def main():

    neurosity = await initialise()                    # initialise Crown headset
    maxlen = 32
    queue = asyncio.Queue(maxsize=maxlen)         # create sliding window (queue)

    # load saved model & spatial filters
    model_file = joblib.load('models/lda_2024-07-21 13:37:50.903718.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # create tasks
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(eeg_stream(queue, neurosity))
        task2 = tg.create_task(eeg_processing(queue, maxlen, model, W))

    # run tasks
    await asyncio.gather(task1, task2)


if __name__ == '__main__':
    asyncio.run(main())




