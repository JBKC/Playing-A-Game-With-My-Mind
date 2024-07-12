'''
Creates motor imagery EEG training data for classification using on-screen prompts
'''

from neurosity import NeurositySDK
from dotenv import load_dotenv
import json
import os
from datetime import datetime
import time
import pygame
import random

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

    # initialise prompt window
    pygame.init()
    pygame.font.init()
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()


    # training parameters
    tasks = ["Right arm - imagine picking up a mug", "Left arm - imagine picking up a mug"]
    interval = 4            # interval between prompts in seconds
    next_prompt_time = time.time() + interval
    current_task = ""
    timestamps = []

    time.sleep(1)


    def callback(data):
        global iter, complete
        stream.append(data)
        iter += 1
        print(f'iter: {iter}')
        if iter >= 100:
            complete = True
            unsubscribe()

    def display_text(text, color=(255,255,255)):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(width / 2, height / 2))
        screen.blit(text_surface, text_rect)

    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)

    while not complete:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                complete = True

        screen.fill((0,0,0))

        current_time = time.time()
        if current_time >= next_prompt_time:
            current_task = random.choice(tasks)
            timestamps.append((current_time, current_task))
            next_prompt_time = current_time + interval

        if current_task:
            display_text(current_task)
        else:
            display_text("Get ready...")

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()
    def process_and_save_data(stream, timestamps):
        # Here you would process your EEG data and align it with the timestamps
        # For example:
        for stamp in timestamps:
            print(f"Time: {stamp[0]}, Task: {stamp[1]}")

    process_and_save_data(stream, timestamps)


    # save stream
    folder = f'test data'
    os.makedirs(folder, exist_ok=True)
    result = [item.to_dict() if hasattr(item, 'to_dict') else item for item in stream]

    with open(f'{folder}/test_{datetime.now()}.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()

