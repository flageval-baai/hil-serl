from pyrobotiqgripper import RobotiqGripper
import threading

class RobotiqGripperServer(RobotiqGripper):
    def __init__(self):
        super().__init__()
        # Use RLock (reentrant lock) because open()/close() call goTo() internally
        self._serial_lock = threading.RLock()
        self._cached_gripper_pos = 0.0  # Store last known good position

    def readAll(self):
        """Override to update cached position after every read"""
        super().readAll()
        # Update cached position every time readAll succeeds
        gPO = self.paramDic.get("gPO")
        if gPO is not None:
            self._cached_gripper_pos = 1 - gPO / 255

    @property
    def gripper_pos(self):
        # Try to read fresh position, but fall back to cached data if serial is busy
        # This prevents conflicts when a gripper command is running
        try:
            if self._serial_lock.acquire(blocking=False):
                try:
                    self.readAll()
                finally:
                    self._serial_lock.release()
            # Always return cached position (updated by readAll, avoids race condition)
            return self._cached_gripper_pos
        except Exception:
            return self._cached_gripper_pos

    def move(self, position, speed=255, force=255):
        """Move gripper to position (non-blocking, just sends command)"""
        position = int(position)
        speed = int(speed)
        force = int(force)

        with self._serial_lock:
            # Send move command without waiting for completion
            # rACT=1 (Activate) and rGTO=1 (Go to Position Request)
            self.write_registers(1000, [0b0000100100000000,
                                        position,
                                        speed * 0b100000000 + force])

    def open(self, speed=255, force=255):
        """Open the gripper (blocking)"""
        with self._serial_lock:
            super().open(speed, force)

    def close(self, speed=255, force=255):
        """Close the gripper (blocking)"""
        with self._serial_lock:
            super().close(speed, force)

    def goTo(self, position, speed=255, force=255):
        """Go to position (blocking)"""
        with self._serial_lock:
            return super().goTo(position, speed, force)