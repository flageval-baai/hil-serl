import numpy as np
from absl import app, flags
import time
import cv2

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "spacemouse_teleop", "Name of experiment corresponding to folder.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment()
    
    env.reset()
    print("Reset done")
    
    while True:
        actions = np.zeros(env.action_space.sample().shape) 
        env.step(actions)
        time.sleep(0.01)
    
        
if __name__ == "__main__":
    app.run(main)