import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    HumanClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.spacemouse_teleop.wrapper import TELEOPEnv


# class EnvConfig(DefaultEnvConfig):
#     SERVER_URL = "http://127.0.0.2:5000/"  # 匹配右臂服务器端口

#     # 临时禁用 ZED 摄像头配置 - 因为 pyzed 模块无法正常工作
#     # 实际可用的摄像头序列号：
#     # ZED-M: 15599084
#     # ZED 2: 26360181  
#     # ZED 2: 22261540
    
#     # 使用空配置来绕过摄像头初始化问题，专注于测试其他功能
#     ZED_CAMERAS = {
#         "wrist": {
#             "serial_number": "14463309",
#             "dim": (1280, 720),
#             "exposure": 40000,
#         },
#         "side_right": {
#             "serial_number": "38232451",
#             "dim": (1280, 720),
#             "exposure": 40000,
#         },
#         # "side_left": {
#         #     "serial_number": "38232451",
#         #     "dim": (1280, 720),
#         #     "exposure": 40000,
#         # },
#     }

#     # 图像裁剪配置
#     IMAGE_CROP = {
#         "zed_m": lambda img: img[100:-100, 200:-200],      # ZED-M 摄像头裁剪
#         "zed2_1": lambda img: img[150:-150, 300:-300],     # ZED 2 摄像头1裁剪
#         "zed2_2": lambda img: img[150:-150, 300:-300],     # ZED 2 摄像头2裁剪
#     }

#     # RESET_POSE = np.array(
#     #     [
#     #         0.5785225644878969,0.0222135215493489,0.18629064082691618,
#     #         3.1405456287466036,
#     #         0.012270296830370064,
#     #         0.002389949492528771,
#     #     ]
#     # )  # 初始位姿 向下
#     RESET_POSE = np.array(
#         [
#             0.4785225644878969,0.0222135215493489,0.60629064082691618,
#             3.1405456287466036,
#             0.012270296830370064,
#             0.002389949492528771,
#         ]
#     )  # 初始位姿 向下
#     # RESET_POSE = np.array([0.3913628586687851,0.043325557670220866,0.602417345242733,3.12939075238998,0.30130763204561596,0.004790074951020358]) # 初始位姿 向前
#     ACTION_SCALE = (0.04, 0.1, 1)  # 映射scale
#     ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])
#     ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])
#     # 修改记录时间 
#     MAX_EPISODE_LENGTH = 200
    

#     def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
#         env = TELEOPEnv(
#             config=EnvConfig(),
#         )
#         if not fake_env:
#             env = SpacemouseIntervention(env)
#         env = RelativeFrame(env)
#         env = Quat2EulerWrapper(env)
#         env = HumanClassifierWrapper(env)
#         return env

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    RESET_POSE = np.array([0.5671124922989944,6.47218270568564e-05,0.4951717570264977,3.1406597193535584,-0.06601965456071524,4.5924120475993035e-05]) # 初始位姿 向下
    # RESET_POSE = np.array([0.3913628586687851,0.043325557670220866,0.602417345242733,3.12939075238998,0.30130763204561596,0.004790074951020358]) # 初始位姿 向前
    ACTION_SCALE = (0.04, 0.1, 1) # 映射scale
    ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])

    def get_environment(self):
        env = TELEOPEnv(
            config=EnvConfig(),
        )
        env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        return env