'''
File to map EEG kinetic thought training to button outputs
'''

from neurosity import NeurositySDK
import os
from dotenv import load_dotenv
import time

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

def training_setup(neurosity):
    metric = "kinesis"
    label = "leftArm"

    training_options = {
        "metric": metric,
        "label": label,
        "experimentId": "-training-1"
    }

    # set up kinesis training + predictions
    def kinesis_callback(kinesis):
        print("leftArm kinesis detection", kinesis)
    def predictions_callback(prediction):
        print("leftArm prediction", prediction)

    neurosity.kinesis(label).subscribe(kinesis_callback)
    neurosity.predictions(label).subscribe(predictions_callback)

    # Tell the user to clear their mind
    print("Clear your mind and relax")

    # Tag baseline after a couple seconds
    time.sleep(4)
    neurosity.training.record({**training_options, "baseline": True})

    # Now tell the user to imagine an active thought
    print("Imagine a baseball with your left arm")

    # Tell the user to imagine active thought and fit
    time.sleep(4)
    neurosity.training.record({**training_options, "fit": True})


def main():
    neurosity = initialise()
    training_setup(neurosity)

if __name__ == '__main__':
    main()

