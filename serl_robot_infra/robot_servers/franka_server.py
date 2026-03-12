"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import rospy
import time
import subprocess
import threading
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R
from absl import app, flags
from typing import Optional

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_msgs.srv import SetLoad
from serl_franka_controllers.msg import ZeroJacobian, DesiredState
import geometry_msgs.msg as geom_msg
from std_msgs.msg import Float64MultiArray
from dynamic_reconfigure.client import Client as ReconfClient


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.3", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", 
    "127.0.0.1",
    "URL for the flask server to run on."
)
flags.DEFINE_string("ros_port", "11311", "Port for the ROS master to run on.")


class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = [float(x) for x in reset_joint_target]
        self.gripper_type = gripper_type

        self._state_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._state_ready = threading.Event()
        self._resetting = threading.Event()
        self._control_mode = "eef"  # "eef" or "joint"

        with self._state_lock:
            self.pos = np.zeros((7,), dtype=np.float64)
            self.vel = np.zeros((6,), dtype=np.float64)
            self.force = np.zeros((3,), dtype=np.float64)
            self.torque = np.zeros((3,), dtype=np.float64)
            self.q = np.zeros((7,), dtype=np.float64)
            self.dq = np.zeros((7,), dtype=np.float64)
            self.jacobian = np.zeros((6, 7), dtype=np.float64)
            self.q_d = np.zeros((7,), dtype=np.float64)    # desired joint positions
            self.pos_d = np.zeros((7,), dtype=np.float64)  # desired EE pose [x,y,z,qx,qy,qz,qw]
            # Unwrapped Euler angles (continuous, no 2π jumps)
            self._euler = np.zeros(3, dtype=np.float64)
            self._euler_d = np.zeros(3, dtype=np.float64)
            self._euler_init = False
            self._euler_d_init = False

        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.joint_pub = rospy.Publisher(
            "/joint_position_controller/command",
            Float64MultiArray,
            queue_size=10,
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        self.desired_state_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/desired_state",
            DesiredState,
            self._set_desired_state,
        )
        time.sleep(1)
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", FrankaState, self._set_currpos
        )

    @staticmethod
    def _unwrap_euler(new_euler, prev_euler):
        """Unwrap Euler angles to maintain continuity with previous values.
        Adjusts new_euler by multiples of 2π so it's closest to prev_euler."""
        diff = new_euler - prev_euler
        return prev_euler + (diff - np.round(diff / (2 * np.pi)) * (2 * np.pi))

    @contextmanager
    def _try_command(self):
        if self._resetting.is_set():
            yield False
            return
        acquired = self._command_lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                self._command_lock.release()

    def is_resetting(self) -> bool:
        return self._resetting.is_set()

    def start_reset_joint_async(self) -> bool:
        if self._resetting.is_set():
            return False

        def _job():
            self._resetting.set()
            try:
                self.reset_joint()
            finally:
                self._resetting.clear()

        threading.Thread(target=_job, name="franka_joint_reset", daemon=True).start()
        return True

    def start_hold_pose_async(
        self,
        pose: np.ndarray,
        *,
        max_s: float = 8.0,
        hz: float = 20.0,
        pos_tol: float = 0.01,
        rot_tol_rad: float = 0.2,
    ) -> bool:
        if self._resetting.is_set():
            return False

        pose = np.asarray(pose, dtype=np.float64).reshape((7,))

        def _job():
            self._resetting.set()
            try:
                with self._command_lock:
                    self.clear()
                    self.wait_for_state(timeout_s=2.0)
                    start_t = time.monotonic()
                    period_s = 1.0 / max(hz, 1e-3)
                    goal_pos = pose[:3]
                    goal_q = pose[3:]
                    while (time.monotonic() - start_t) < max_s:
                        self.move(pose.tolist())
                        reached = False
                        if self._state_ready.is_set():
                            with self._state_lock:
                                curr = np.array(self.pos, copy=True)
                            pos_err = float(np.linalg.norm(curr[:3] - goal_pos))
                            dot = float(np.clip(np.abs(np.dot(curr[3:], goal_q)), -1.0, 1.0))
                            rot_err = float(2.0 * np.arccos(dot))
                            reached = (pos_err <= pos_tol) and (rot_err <= rot_tol_rad)
                        if reached:
                            break
                        time.sleep(period_s)
            finally:
                self._resetting.clear()

        threading.Thread(target=_job, name="franka_hold_pose", daemon=True).start()
        return True

    def start_impedance(self):
        """Launches the impedance controller"""
        with self._command_lock:
            self.imp = subprocess.Popen(
                [
                    "roslaunch",
                    self.ros_pkg_name,
                    "impedance.launch",
                    "robot_ip:=" + self.robot_ip,
                    f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
                ],
                stdout=subprocess.PIPE,
            )
            time.sleep(3)

    def stop_impedance(self):
        """Stops the impedance controller"""
        with self._command_lock:
            proc = getattr(self, "imp", None)
            if proc is None:
                return
            try:
                proc.terminate()
            except Exception:
                pass
            time.sleep(1)
            self.imp = None

    def start_joint_controller(self):
        """Launches the joint position controller"""
        with self._command_lock:
            # Set target joint positions to current positions (required by init)
            with self._state_lock:
                current_q = self.q.tolist()
            rospy.set_param("/target_joint_positions", current_q)

            self.joint_controller_proc = subprocess.Popen(
                [
                    "roslaunch",
                    self.ros_pkg_name,
                    "joint.launch",
                    "robot_ip:=" + self.robot_ip,
                    f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
                ],
                stdout=subprocess.PIPE,
            )
            time.sleep(3)

    def stop_joint_controller(self):
        """Stops the joint position controller"""
        with self._command_lock:
            proc = getattr(self, "joint_controller_proc", None)
            if proc is None:
                return
            try:
                proc.terminate()
            except Exception:
                pass
            time.sleep(1)
            self.joint_controller_proc = None

    def switch_to_joint_mode(self):
        """Switch from impedance to joint position control."""
        if self._control_mode == "joint":
            return
        print("Switching to joint control mode...")
        self.stop_impedance()
        self.clear()
        time.sleep(1)
        self.start_joint_controller()
        self._control_mode = "joint"
        print("Joint control mode active")

    def switch_to_eef_mode(self):
        """Switch from joint position to impedance (EEF) control."""
        if self._control_mode == "eef":
            return
        print("Switching to EEF control mode...")
        self.stop_joint_controller()
        self.clear()
        time.sleep(1)
        self.start_impedance()
        self._control_mode = "eef"
        print("EEF control mode active")

    def clear(self):
        """Clears any errors"""
        with self._command_lock:
            msg = ErrorRecoveryActionGoal()
            self.resetpub.publish(msg)

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        with self._command_lock:
            # First Stop impedance
            try:
                self.stop_impedance()
                self.clear()
            except Exception:
                print("impedance Not Running")
            time.sleep(3)
            self.clear()

            # Launch joint controller reset
            # set rosparm with rospkg
            # rosparam set /target_joint_positions '[q1, q2, q3, q4, q5, q6, q7]'
            rospy.set_param("/target_joint_positions", self.reset_joint_target)

            self.joint_controller = subprocess.Popen(
                [
                    "roslaunch",
                    self.ros_pkg_name,
                    "joint.launch",
                    "robot_ip:=" + self.robot_ip,
                    f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
                ],
                stdout=subprocess.PIPE,
            )
            time.sleep(1)
            print("RUNNING JOINT RESET")
            self.clear()

            # Wait until target joint angles are reached
            count = 0
            time.sleep(1)
            while True:
                with self._state_lock:
                    q = np.array(self.q, copy=True)
                if np.allclose(
                    np.array(self.reset_joint_target) - q,
                    0,
                    atol=1e-2,
                    rtol=1e-2,
                ):
                    break
                time.sleep(1)
                count += 1
                if count > 30:
                    print("joint reset TIMEOUT")
                    break

            # Stop joint controller
            print("RESET DONE")
            try:
                self.joint_controller.terminate()
            except Exception:
                pass
            time.sleep(1)
            self.clear()
            with self._state_lock:
                pos = np.array(self.pos, copy=True)
            print("KILLED JOINT RESET", pos)

            # Restart impedece controller
            self.start_impedance()
            print("impedance STARTED")

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        with self._command_lock:
            assert len(pose) == 7
            msg = geom_msg.PoseStamped()
            msg.header.frame_id = "0"
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
            msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
            self.eepub.publish(msg)

    def move_joints(self, q: np.ndarray):
        """Moves to joint positions: [q1, q2, q3, q4, q5, q6, q7]"""
        with self._command_lock:
            assert len(q) == 7
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)

    def wait_for_state(self, timeout_s: Optional[float] = None) -> bool:
        return self._state_ready.wait(timeout=timeout_s)

    def get_state_copy(self) -> dict:
        with self._state_lock:
            return {
                "pose": np.array(self.pos, copy=True),
                "vel": np.array(self.vel, copy=True),
                "force": np.array(self.force, copy=True),
                "torque": np.array(self.torque, copy=True),
                "q": np.array(self.q, copy=True),
                "dq": np.array(self.dq, copy=True),
                "jacobian": np.array(self.jacobian, copy=True),
                "q_d": np.array(self.q_d, copy=True),
                "pose_d": np.array(self.pos_d, copy=True),
                "euler": np.array(self._euler, copy=True),
                "euler_d": np.array(self._euler_d, copy=True),
            }

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        euler = r.as_euler("xyz")
        dq = np.array(list(msg.dq)).reshape((7,))
        q = np.array(list(msg.q)).reshape((7,))
        force = np.array(list(msg.K_F_ext_hat_K)[:3])
        torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        with self._state_lock:
            self.pos = pose
            if self._euler_init:
                self._euler = self._unwrap_euler(euler, self._euler)
            else:
                self._euler = euler
                self._euler_init = True
            self.dq = dq
            self.q = q
            self.force = force
            self.torque = torque
            try:
                self.vel = self.jacobian @ self.dq
            except Exception:
                self.vel = np.zeros(6)
                rospy.logwarn(
                    "Jacobian not set, end-effector velocity temporarily not available"
                )
            self._state_ready.set()

    def _set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        with self._state_lock:
            self.jacobian = jacobian

    def _set_desired_state(self, msg):
        q_d = np.array(list(msg.q_d)).reshape((7,))
        pose_d = np.array(list(msg.pose_d)).reshape((7,))
        euler_d = R.from_quat(pose_d[3:]).as_euler("xyz")
        with self._state_lock:
            self.q_d = q_d
            self.pos_d = pose_d
            if self._euler_d_init:
                self._euler_d = self._unwrap_euler(euler_d, self._euler_d)
            else:
                self._euler_d = euler_d
                self._euler_d_init = True


###############################################################################


def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"

    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen(f"roscore -p {FLAGS.ros_port}", shell=True)
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    # Start ros node
    rospy.init_node("franka_control_api")

    if GRIPPER_TYPE == "Robotiq":
        if GRIPPER_IP.startswith("0"):
            from robot_servers.pyrobotiq_gripper_server import RobotiqGripperServer
            gripper_server = RobotiqGripperServer()
        else:
            from robot_servers.robotiq_gripper_server import RobotiqGripperServer

            gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)

    elif GRIPPER_TYPE == "Franka":
        from robot_servers.franka_gripper_server import FrankaGripperServer

        gripper_server = FrankaGripperServer()
    elif GRIPPER_TYPE == "None":
        gripper_server = None
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    """Starts impedance controller"""
    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    robot_server.start_impedance()

    reconf_client = ReconfClient(
        "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
    )

    rospy.wait_for_service('/franka_control/set_load')
    set_load_service = rospy.ServiceProxy('/franka_control/set_load', SetLoad)

    # Lock to prevent concurrent gripper commands
    gripper_command_lock = threading.Lock()

    def _busy(action: str):
        return jsonify({
            "ok": False,
            "busy": True,
            "action": action,
            "resetting": robot_server.is_resetting(),
        })

    def _get_gripper_position() -> float:
        """Get current gripper position (non-blocking)."""
        if gripper_server is None:
            return 0.0
        return float(getattr(gripper_server, "gripper_pos", 0.0))

    def _get_gripper_desired_position() -> float:
        """Get desired gripper position from hardware register (gPR).
        Falls back to actual position if not available."""
        if gripper_server is None:
            return 0.0
        pos_d = getattr(gripper_server, "gripper_pos_d", None)
        if pos_d is None:
            return _get_gripper_position()
        return float(pos_d)


    def _start_gripper_command_async(command_fn, *args, **kwargs) -> bool:
        """Start a gripper command in a background thread. Returns False if busy."""
        if not gripper_command_lock.acquire(blocking=False):
            return False

        def _job():
            try:
                command_fn(*args, **kwargs)
            finally:
                gripper_command_lock.release()

        threading.Thread(target=_job, name="gripper_command", daemon=True).start()
        return True

    def _gripper_route(action: str, method_name: str, *args):
        """Helper for gripper routes - handles common checks and async execution."""
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy(action)
        command_fn = getattr(gripper_server, method_name)
        if not _start_gripper_command_async(command_fn, *args):
            return _busy(action)
        return jsonify({"ok": True})


    # Route for Setting Load
    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("set_load")
        data = request.json
        mass = data['mass']
        F_x_center_load = data['F_x_center_load']
        load_inertia = data['load_inertia']
        set_load_service(mass, F_x_center_load, load_inertia)
        print("Set mass to", mass)
        return jsonify({"ok": True})

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("startimp")
            robot_server.clear()
            robot_server.start_impedance()
            return jsonify({"ok": True})

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("stopimp")
            robot_server.stop_impedance()
            return jsonify({"ok": True})
    
    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        robot_server.wait_for_state(timeout_s=2.0)
        state = robot_server.get_state_copy()
        xyz = state["pose"][:3]
        return jsonify({"pose": np.concatenate([xyz, state["euler"]]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"pose": robot_server.get_state_copy()["pose"].tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"vel": robot_server.get_state_copy()["vel"].tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"force": robot_server.get_state_copy()["force"].tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"torque": robot_server.get_state_copy()["torque"].tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"q": robot_server.get_state_copy()["q"].tolist()})

    @webapp.route("/getq_d", methods=["POST"])
    def get_q_d():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"q_d": robot_server.get_state_copy()["q_d"].tolist()})

    @webapp.route("/getpos_euler_d", methods=["POST"])
    def get_pos_euler_d():
        robot_server.wait_for_state(timeout_s=2.0)
        state = robot_server.get_state_copy()
        xyz = state["pose_d"][:3]
        return jsonify({"pose_d": np.concatenate([xyz, state["euler_d"]]).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"dq": robot_server.get_state_copy()["dq"].tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        robot_server.wait_for_state(timeout_s=2.0)
        return jsonify({"jacobian": robot_server.get_state_copy()["jacobian"].tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": _get_gripper_position()})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        started = robot_server.start_reset_joint_async()
        if not started:
            return _busy("jointreset")
        return jsonify({"ok": True, "started": True})

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        return _gripper_route("activate_gripper", "activate_gripper")

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        return _gripper_route("reset_gripper", "reset_gripper")

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        print("open gripper")
        return _gripper_route("open_gripper", "open")

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        print("close gripper")
        return _gripper_route("close_gripper", "close")

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        print("close gripper slow")
        return _gripper_route("close_gripper_slow", "close_slow")

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = np.clip(int(request.json["gripper_pos"]), 0, 255)
        print(f"move gripper to {pos}")
        return _gripper_route("move_gripper", "move", pos)

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("clearerr")
            robot_server.clear()
            return jsonify({"ok": True})

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("pose")
            robot_server.move(pos)
            return jsonify({"ok": True})

    # Route for Sending joint position command
    @webapp.route("/joints", methods=["POST"])
    def joints():
        q = np.array(request.json["arr"], dtype=np.float64)
        if len(q) != 7:
            return jsonify({"ok": False, "error": "expected 7 joint positions"}), 400
        if robot_server._control_mode != "joint":
            return jsonify({"ok": False, "error": "not in joint mode, call /start_joint_control first"}), 400
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("joints")
            robot_server.move_joints(q)
            return jsonify({"ok": True})

    # Route for switching to joint control mode
    @webapp.route("/start_joint_control", methods=["POST"])
    def start_joint_control():
        if robot_server._control_mode == "joint":
            return jsonify({"ok": True, "mode": "joint", "msg": "already in joint mode"})
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("start_joint_control")
            robot_server.switch_to_joint_mode()
            return jsonify({"ok": True, "mode": "joint"})

    # Route for switching back to EEF (impedance) control mode
    @webapp.route("/start_eef_control", methods=["POST"])
    def start_eef_control():
        if robot_server._control_mode == "eef":
            return jsonify({"ok": True, "mode": "eef", "msg": "already in eef mode"})
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("start_eef_control")
            robot_server.switch_to_eef_mode()
            return jsonify({"ok": True, "mode": "eef"})

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        robot_server.wait_for_state(timeout_s=2.0)
        state = robot_server.get_state_copy()
        return jsonify(
            {
                "pose": state["pose"].tolist(),
                "vel": state["vel"].tolist(),
                "force": state["force"].tolist(),
                "torque": state["torque"].tolist(),
                "q": state["q"].tolist(),
                "dq": state["dq"].tolist(),
                "jacobian": state["jacobian"].tolist(),
                "gripper_pos": _get_gripper_position(),
                "gripper_pos_d": _get_gripper_desired_position(),
                "pose_euler": np.concatenate([
                    state["pose"][:3],
                    state["euler"],
                ]).tolist(),
                "q_d": state["q_d"].tolist(),
                "pose_euler_d": np.concatenate([
                    state["pose_d"][:3],
                    state["euler_d"],
                ]).tolist(),
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("update_param")
            reconf_client.update_configuration(request.json)
            return jsonify({"ok": True})

    @webapp.route("/status", methods=["POST"])
    def status():
        return jsonify(
            {
                "ok": True,
                "resetting": robot_server.is_resetting(),
                "state_ready": robot_server._state_ready.is_set(),
                "control_mode": robot_server._control_mode,
            }
        )
    
    JOINT_RESET_Q = [0, 0, 0, -1.9, 0, 2, 0]
    EEF_RESET_POSE = [0.5671124922989944, 6.47218270568564e-05, 0.4951717570264977,
                      -3.1281813611027127, 0.06279893041386786, 0.00925549227719924]

    @webapp.route("/reset_all", methods=["POST"])
    def reset_all():
        data = request.get_json(silent=True) or {}

        if gripper_server is not None:
            _start_gripper_command_async(gripper_server.open)

        if robot_server._control_mode == "joint":
            # Joint mode: send reset joint positions via streaming command
            target = np.array(JOINT_RESET_Q, dtype=np.float64)
            robot_server.move_joints(target)
            # Wait for convergence
            tol = float(data.get("tol", 0.01))
            timeout = float(data.get("timeout", 10.0))
            t0 = time.time()
            converged = False
            while time.time() - t0 < timeout:
                time.sleep(0.1)
                state = robot_server.get_state_copy()
                q = state["q"]
                if max(abs(c - t) for c, t in zip(q, JOINT_RESET_Q)) < tol:
                    converged = True
                    break
            return jsonify({"ok": True, "converged": converged,
                            "mode": "joint", "target": JOINT_RESET_Q})
        else:
            # EEF mode: hold pose async
            max_s = float(data.get("max_s", 10.0))
            hz = float(data.get("hz", 30.0))
            pos_tol = float(data.get("pos_tol", 0.01))
            rot_tol_rad = float(data.get("rot_tol_rad", 0.2))
            goal = np.array(EEF_RESET_POSE)
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
            pos = goal.astype(np.float32)
            started = robot_server.start_hold_pose_async(
                pos, max_s=max_s, hz=hz, pos_tol=pos_tol, rot_tol_rad=rot_tol_rad
            )
            if not started:
                return _busy("reset_all")
            return jsonify({"ok": True, "started": True, "mode": "eef"})

    webapp.run(host=FLAGS.flask_url, threaded=True)


if __name__ == "__main__":
    app.run(main)
