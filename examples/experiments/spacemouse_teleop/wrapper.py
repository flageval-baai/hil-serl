import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard

from franka_env.envs.franka_env import FrankaEnv

class TELEOPEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_reset = False
        self.joint_reset = False

        def on_press(key):
            if str(key) == "Key.r":
                self.should_reset = True
            elif str(key) == "Key.j":
                self.joint_reset = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

    def go_to_reset(self):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        # requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        # perform joint reset if needed
        if self.joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        reset_pose = self.resetpos.copy()
        self._send_pos_command(reset_pose)
        self._send_gripper_command(1.0)
        time.sleep(0.5)

        # Change to compliance mode
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
    
    def reset(self, **kwargs):

        self._recover()
        self.go_to_reset()
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}