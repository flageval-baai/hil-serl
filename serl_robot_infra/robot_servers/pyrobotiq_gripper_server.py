from pyrobotiqgripper import RobotiqGripper

class RobotiqGripperServer(RobotiqGripper):
    def __init__(self):
        super().__init__()
    
    @property
    def gripper_pos(self):
        return 1 - self.getPosition() / 255