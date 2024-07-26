'''
Stream EEG data and run realtime inference on trained model
Version 2 - uses a single deque to connect producer + consumer (may lead to data loss)
'''

from neurosity import NeurositySDK
import asyncio
from dotenv import load_dotenv
import numpy as np
import os
import crown_realtime_processing
import joblib
import collections
import pygame
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

def eeg_stream_callback(window):
    def callback(data):
        # retrieve new data
        signals = np.array(data['data'])
        # add data to sliding window
        window.append(signals)

        print(f'Window size: {len(window)}')

    return callback

# coroutine 1: pull EEG data
async def eeg_stream(window, neurosity):

    callback = eeg_stream_callback(window)
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)                 # neurosity expects synchronous callback

    try:
        while True:
            # yield control back to event loop after callback has been executed
            await asyncio.sleep(0)

    # stop stream when program exits
    finally:
        unsubscribe()

# coroutine 2: process EEG data
async def eeg_processing(window, maxlen, model, W, probs_queue):

    while True:
        # start processing if window is at maximum length
        if len(window) == maxlen:

            probs = await asyncio.to_thread(crown_realtime_processing.main,
                                    np.array(window).transpose(1, 0, 2).reshape(8, -1), model, W)

            await probs_queue.put(probs)


async def pygame_display(probs_queue):
    '''
    :param: probs_queue: queue containing arrays of length 2: [right probability, left probability]
    '''

    pygame.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("EEG Probability Display")
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    BLACK, WHITE, RED, GREEN = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)

    probs = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # check for updates in the queue
        if not probs_queue.empty():
            probs = await probs_queue.get()

        screen.fill(BLACK)

        if probs is not None:

            right, left = probs
            # render text
            right_text = font.render(f"Right: {right:.2f}", True, WHITE)
            left_text = font.render(f"Left: {left:.2f}", True, WHITE)
            # draw text
            screen.blit(right_text, (50, 200))
            screen.blit(left_text, (50, 250))
            # create dynamic bar for each class probability
            pygame.draw.rect(screen, RED, (400, 200, int(right * 300), 30))
            pygame.draw.rect(screen, GREEN, (400, 250, int(left * 300), 30))
        else:
            text = font.render("Initialising...", True, WHITE)
            screen.blit(text, (width // 2 - text.get_width() // 2, height // 2))

        pygame.display.flip()
        clock.tick(30)
        await asyncio.sleep(0)  # yield control to the event loop

    pygame.quit()


async def main():

    neurosity = await initialise()                    # initialise Crown headset
    maxlen = 32
    window = collections.deque(maxlen=maxlen)         # create sliding window (queue)
    probs_queue = asyncio.Queue()                     # queue for right/left probabilities

    # load saved model & spatial filters
    model_file = joblib.load('models/lda_2024-07-21 13:37:50.903718.joblib')
    model = model_file['model']
    W = model_file['spatial_filters']

    # create tasks
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(eeg_stream(window, neurosity))
        task2 = tg.create_task(eeg_processing(window, maxlen, model, W, probs_queue))
        task3 = tg.create_task(pygame_display(probs_queue))

    # run tasks
    await asyncio.gather(task1, task2, task3)

    # run tasks
    await asyncio.gather(task1, task2)



if __name__ == '__main__':
    asyncio.run(main())





