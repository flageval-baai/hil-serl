import numpy as np
from copy import deepcopy

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")


class ZedCapture:
    def get_device_serial_numbers(self):
        try:
            cameras = sl.Camera.get_device_list()
            return [str(cam.serial_number) for cam in cameras]
        except NameError:
            return []

    def __init__(
        self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000
    ):
        self.name = name
        print(f"尝试连接 ZED 摄像头: {name}")
        print(f"指定序列号: {serial_number}")
        
        available_devices = self.get_device_serial_numbers()
        print(f"检测到的设备: {available_devices}")
        
        if not available_devices:
            print("⚠️  警告: 没有检测到任何 ZED 设备")
            print("可能的原因:")
            print("1. ZED SDK 没有正确安装")
            print("2. pyzed 模块有问题")
            print("3. 摄像头没有正确连接")
            print("4. 设备权限问题")
            raise RuntimeError(f"无法检测到任何 ZED 摄像头设备。请检查 ZED SDK 安装和摄像头连接。")
        
        if str(serial_number) not in available_devices:
            print(f"❌ 错误: 找不到序列号为 {serial_number} 的设备")
            print(f"可用设备序列号: {available_devices}")
            raise ValueError(f"找不到序列号为 {serial_number} 的 ZED 摄像头。可用设备: {available_devices}")
            
        print(f"✅ 找到匹配的设备，序列号: {serial_number}")
        self.serial_number = serial_number
        self.depth = depth

        # Initialize ZED camera using official interface from zed_camera.py
        self._cam = sl.Camera()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._depth_img = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Set initialization parameters based on zed_camera.py standard_params
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(serial_number))
        init_params.camera_resolution = self._get_resolution_from_dim(dim)
        init_params.camera_fps = fps
        # init_params.depth_minimum_distance = 0.1
        # init_params.depth_stabilization = False
        init_params.camera_image_flip = sl.FLIP_MODE.OFF

        status = self._cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")

        # Store resolution for frame processing
        self.zed_resolution = sl.Resolution(*dim)

    def _get_resolution_from_dim(self, dim):
        width, height = dim
        if width <= 672 and height <= 376:
            return sl.RESOLUTION.VGA
        elif width <= 1280 and height <= 720:
            return sl.RESOLUTION.HD720
        elif width <= 1920 and height <= 1080:
            return sl.RESOLUTION.HD1080
        elif width <= 2208 and height <= 1242:
            return sl.RESOLUTION.HD2K
        else:
            return sl.RESOLUTION.HD720

    def read(self):
        # Read camera using official interface from zed_camera.py
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return False, None

        # Retrieve left image (main image) - following zed_camera.py pattern
        self._cam.retrieve_image(
            self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution
        )
        image = deepcopy(self._left_img.get_data())

        if self.depth:
            self._cam.retrieve_measure(
                self._depth_img, sl.MEASURE.DEPTH, resolution=self.zed_resolution
            )
            depth = deepcopy(self._depth_img.get_data())
            depth = np.expand_dims(depth, axis=2)
            return True, np.concatenate((image, depth), axis=-1)
        else:
            return True, image

    def close(self):
        if hasattr(self, "_cam"):
            self._cam.close()
