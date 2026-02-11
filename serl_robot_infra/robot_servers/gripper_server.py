class GripperServer:
    def __init__(self):
        self.gripper_pos = 0
        self.gripper_pos_d = None  # desired position; None = use gripper_pos

    def open(self):
        pass

    def close(self):
        pass

    def move(self, position: int):
        pass

    def activate_gripper(self):
        pass

    def reset_gripper(self):
        pass
