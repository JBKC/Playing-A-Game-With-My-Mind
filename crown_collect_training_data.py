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

    # initialise prompt window
    pygame.init()
    pygame.font.init()
    width, height = 600, 600
    # os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"  # Position window at top-left
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    pygame.display.set_allow_screensaver(False)
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # training parameters
    tasks = {
        "action": ["Right arm - imagine curling a dumbbell", "Left arm - imagine curling a dumbbell"],
        "label": [-1, 1],
        "relax": "Relax"
    }

    n_trials = 20                                # set this - number of desired trials for EACH class
    interval = 4                                 # set this - length of each prompt (seconds)

    n_iters = n_trials*4 + 1                     # total number of prompts that will appear (includes relaxation)
    eeg_iters = (n_iters * interval) * 16        # set stopping point for EEG
    next_time = time.time() + interval
    current_task = ""
    timestamps = []

    # pull EEG data
    def callback(data):
        global iter, complete
        stream.append(data)
        iter += 1
        print(f'eeg iter: {iter}')
        if iter >= eeg_iters:
            complete = True
            unsubscribe()

    # begin data callback
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)
    # record start time of prompt window
    timestamps.append((time.time(), "startTime"))

    # display prompt window
    def display_text(text, color=(0, 0, 0)):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(width / 2, height / 2))
        screen.blit(text_surface, text_rect)

    def class_split():
        # create an even class split that comes up in a random order from user's POV
        # 1/4 right, 1/4 left, 1/2 relax

        class1 = [tasks["action"][0]] * (n_iters // 4)              # right arm
        class2 = [tasks["action"][1]] * (n_iters // 4)              # left arm

        # shuffle order
        order = []
        actions = class1 + class2
        random.shuffle(actions)

        for task in actions:
            order.extend([task, tasks["relax"]])                # alternate between action and relax

        return order


    # bring prompt window to front
    pygame.display.set_mode((width, height))
    pygame.display.flip()

    # get randomised prompt order
    prompt_order = class_split()
    prompt_counter = 0

    # iterate through prompts
    while prompt_counter < len(prompt_order):
        screen.fill((255, 255, 255))
        current_time = time.time()

        if current_time >= next_time:
            current_task = prompt_order[prompt_counter]

            # assign labels
            if current_task != "Relax":
                label = tasks["label"][tasks["action"].index(current_task)]
            else:
                label = 0

            # record data
            timestamps.append((current_time, current_task, label))
            next_time = current_time + interval
            prompt_counter += 1

        if current_task:
            display_text(current_task)
        else:
            display_text("Get ready...")

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(10)  # limit to 10 FPS

    # finishing screen
    screen.fill((255, 255, 255))
    display_text("FINISHED. Keep headset on", color=(0, 0, 0))
    pygame.display.flip()
    pygame.quit()
    print(f"Prompts finished; waiting for EEG to finish collecting data")

    # wait for EEG to finish
    while not complete:
        time.sleep(0.1)

    def save_json():
        # save EEG stream + prompt timestamps
        folder = f'training_data'
        os.makedirs(folder, exist_ok=True)

        eeg_stream = [item.to_dict() if hasattr(item, 'to_dict') else item for item in stream]
        label_data = [stamp.to_dict() if hasattr(stamp, 'to_dict') else stamp for stamp in timestamps]

        with open(f'{folder}/eeg_stream_{datetime.now()}.json', 'w') as f:
            json.dump(eeg_stream, f, indent=2)

        with open(f'{folder}/labels_{datetime.now()}.json', 'w') as f:
            json.dump(label_data, f, indent=2)

    save_json()

if __name__ == '__main__':
    main()


