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
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"  # Position window at top-left
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    pygame.display.set_allow_screensaver(False)
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # training parameters
    tasks = {
        "action": ["Right hand - imagine picking up a mug", "Left hand - imagine picking up a mug"],
        "label": [-1, 1],
        "relax": "Relax"
    }
    n_iters = 4
    interval = 4  # length of each prompt (s)
    eeg_iters = (n_iters * interval) * 16  # set stopping point for EEG
    next_time = time.time() + interval
    action_flag = True
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

    # display prompt window
    def display_text(text, color=(0, 0, 0)):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(width / 2, height / 2))
        screen.blit(text_surface, text_rect)

    # begin data callback
    unsubscribe = neurosity.brainwaves_raw_unfiltered(callback)

    def next_prompt(action_flag):
        return random.choice(tasks["action"]) if action_flag else tasks["relax"]

    timestamps.append((time.time(), "startTime"))

    # bring prompt window to front
    pygame.display.set_mode((width, height))
    pygame.display.flip()

    start_time = time.time()

    # iterate through prompts
    while time.time() - start_time < n_iters * interval:
        screen.fill((255, 255, 255))
        current_time = time.time()

        if current_time >= next_time:
            current_task = next_prompt(action_flag)

            if current_task != "Relax":
                label = tasks["label"][tasks["action"].index(current_task)]
            else:
                label = None

            timestamps.append((current_time, label))
            next_time = current_time + interval
            action_flag = not action_flag

        if current_task:
            display_text(current_task)
        else:
            display_text("Get ready...")

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(60)  # limit to 60 FPS

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
        folder = f'test data'
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


