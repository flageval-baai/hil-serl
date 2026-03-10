from pyrobotiqgripper import RobotiqGripper
import threading


class RobotiqGripperServer(RobotiqGripper):
    """Thread-safe Robotiq gripper server with real-time position tracking."""

    # Physical register limits (measure per gripper unit)
    GPO_OPEN = 3      # gPO when fully open
    GPO_CLOSED = 230   # gPO when fully closed (no object)

    def __init__(self):
        super().__init__()
        self._serial_lock = threading.RLock()
        self._cached_gripper_pos = 0.0
        self.gripper_pos_d = None  # None → fall back to gripper_pos

    def _normalize(self, raw):
        """Map raw register value to [0, 1]: 0=closed, 1=open, clipped."""
        return max(0.0, min(1.0,
            (self.GPO_CLOSED - raw) / (self.GPO_CLOSED - self.GPO_OPEN)
        ))

    def readAll(self):
        """Override to update cached position after every read."""
        super().readAll()
        gPO = self.paramDic.get("gPO")
        if gPO is not None:
            self._cached_gripper_pos = self._normalize(gPO)
            # Track gripper_pos_d from gPO (actual position), not gPR.
            # gPR is unreliable because _autoConnect() contaminates it
            # with rPR=100 and readAll() uses FC3 instead of FC4.
            # During goTo() blocking calls, the internal polling loop
            # calls readAll() continuously, keeping this value current.
            self.gripper_pos_d = self._cached_gripper_pos

    @property
    def gripper_pos(self):
        """Get gripper position (0=closed, 1=open). Non-blocking, uses cache if busy."""
        try:
            if self._serial_lock.acquire(blocking=False):
                try:
                    self.readAll()
                finally:
                    self._serial_lock.release()
            return self._cached_gripper_pos
        except Exception:
            return self._cached_gripper_pos

    def goTo(self, position, speed=255, force=255):
        """Go to position (blocking, thread-safe)."""
        with self._serial_lock:
            return super().goTo(position, speed, force)

    def move(self, position, speed=255, force=255):
        """Send move command (non-blocking, thread-safe)."""
        with self._serial_lock:
            self.write_registers(1000, [0b0000100100000000,
                                        int(position),
                                        int(speed) * 256 + int(force)])

    def activate_gripper(self):
        """Activate the gripper (blocking, thread-safe)."""
        with self._serial_lock:
            self.activate()

    def reset_gripper(self):
        """Reset and activate the gripper (blocking, thread-safe)."""
        with self._serial_lock:
            self.resetActivate()

    def close_slow(self, speed=50, force=255):
        """Close the gripper slowly (blocking, thread-safe)."""
        self.goTo(255, speed, force)
