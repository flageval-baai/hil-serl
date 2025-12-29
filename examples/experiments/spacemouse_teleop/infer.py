import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import time
import pyrealsense2 as rs
import cv2
import argparse
import os

class Recorder:
    def __init__(self):

        self.connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        assert len(self.connect_device) == 1
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.connect_device[0])
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def record_frame(self):
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale * 1000
        
        return color_image, depth_image
    

class MultiRecorder:
    def __init__(self):
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))

        assert len(self.connect_device) == 2

        self.pipeline_front = rs.pipeline()
        self.config.enable_device(self.connect_device[1])
        self.pipeline_front.start(self.config)
        self.pipeline_side = rs.pipeline()
        self.config.enable_device(self.connect_device[0])
        self.pipeline_side.start(self.config)

        self.depth_scale = 0.0010000000474974513

    def record_frame(self):

        frames_front = self.pipeline_front.wait_for_frames()
        frames_side = self.pipeline_side.wait_for_frames()
        aligned_frames_front = self.align.process(frames_front)
        aligned_frames_side = self.align.process(frames_side)
    
        color_frame_front = aligned_frames_front.get_color_frame()
        color_frame_side = aligned_frames_side.get_color_frame()
        depth_frame_front = aligned_frames_front.get_depth_frame()
        depth_frame_side = aligned_frames_side.get_depth_frame()
        
        color_image_front = np.asanyarray(color_frame_front.get_data(), dtype=np.uint8)
        color_image_side = np.asanyarray(color_frame_side.get_data(), dtype=np.uint8)
        depth_image_front = np.asanyarray(depth_frame_front.get_data(), dtype=np.float32) * self.depth_scale * 1000
        depth_image_side = np.asanyarray(depth_frame_side.get_data(), dtype=np.float32) * self.depth_scale * 1000
        
        return color_image_front, color_image_side, depth_image_front, depth_image_side

def get_pose_quat():
    url = "http://127.0.0.1:5000/getpos"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_pose_euler():
    url = "http://127.0.0.1:5000/getpos_euler"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_joint():
    url = "http://127.0.0.1:5000/getq"
    response = requests.post(url)
    cur_joint = response.json()['q']
    
    return cur_joint    

def get_gripper():
    url = "http://127.0.0.1:5000/get_gripper"
    response = requests.post(url)
    cur_gripper = response.json()['gripper']
    gripper_open = 1 if cur_gripper > 0.03 else 0
    
    return np.array(gripper_open)

def goto_pose(pose):
    url = "http://127.0.0.1:5000/pose"
    message = {
        "arr": pose
    }
    
    requests.post(url, json=message)
    
def goto_gripper(gripper):
    if gripper == 1:
        message = {"gripper_pos": 255}
    else:
        message = {"gripper_pos": 20}
    
    url = "http://127.0.0.1:5000/move_gripper"
    requests.post(url, json=message)

def load_model():
    
    raise NotImplementedError

def model_infer(args):
    
    raise NotImplementedError 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--task', type=str, default='infer', required=False)
    parser.add_argument('--camera_num', type=int, default=1)
    
    args = parser.parse_args()
    
    output_root = "/home/guchenyang/Codes/hil-serl/examples/experiments/spacemouse_teleop/data"
    output_dir = os.path.join(output_root, args.task, str(args.index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    poses = []
    joints = []
    grippers = []
    
    # model = load_model() # load model
    task_length = 3 # 根据每个任务抽帧来定

    if args.camera_num == 1:
        camera = Recorder()
        # front_images = []
        # front_depths = []
        # front_images_dir = os.path.join(output_dir, "front_images")
        # front_depths_dir = os.path.join(output_dir, "front_depths")
        # os.makedirs(front_images_dir, exist_ok=True)
        # os.makedirs(front_depths_dir, exist_ok=True)
        
        for i in range(task_length):
            # front_image, front_depth = camera.record_frame()
            # image_filename = os.path.join(front_images_dir, f"{i}.png")
            # cv2.imwrite(image_filename, front_images[i])
            # depth_filename = os.path.join(front_depths_dir, f"{i}.npy")
            # np.save(depth_filename, front_depths[i])
            
            joint = get_joint()
            pose = get_pose_euler()
            print(pose)
            gripper = get_gripper()
            print(gripper)
            pose.append(float(gripper))
            print(pose)
    
            # action = model_infer(model, xxx) # 看着改吧，看看输入啥
            
            # target_pose, target_gripper = process_action(action, pose, xxx) # 也看着改，根据模型输出的action来
            
            # goto_pose(target_pose)
            # time.sleep(1) # 因为这个代码是非阻塞的，所以得手动暂停等待pose执行完，具体sleep多少测一下
            # goto_gripper(target_gripper) 
            
            # poses.append(target_pose)
            # grippers.append(target_gripper)
            
        