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
import os
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R
from absl import app, flags
from typing import Optional

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_msgs.srv import SetLoad
from serl_franka_controllers.msg import ZeroJacobian
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

try:
    import rosgraph
except Exception:
    rosgraph = None

try:
    from controller_manager_msgs.srv import ListControllers, LoadController, SwitchController
    from controller_manager_msgs.srv import SwitchControllerRequest
except Exception:
    ListControllers = None
    LoadController = None
    SwitchController = None
    SwitchControllerRequest = None


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

    CARTESIAN_CONTROLLER = "cartesian_impedance_controller"
    JOINT_STREAM_CONTROLLER = "streaming_joint_impedance_controller"

    def __init__(self, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = [float(x) for x in reset_joint_target]
        self.gripper_type = gripper_type

        self._state_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._state_ready = threading.Event()
        self._resetting = threading.Event()

        with self._state_lock:
            self.pos = np.zeros((7,), dtype=np.float64)
            self.vel = np.zeros((6,), dtype=np.float64)
            self.force = np.zeros((3,), dtype=np.float64)
            self.torque = np.zeros((3,), dtype=np.float64)
            self.q = np.zeros((7,), dtype=np.float64)
            self.dq = np.zeros((7,), dtype=np.float64)
            self.jacobian = np.zeros((6, 7), dtype=np.float64)

        self.control_mode = "cartesian"  # cartesian | joint | stopped

        self._cm_ns = None
        self._cm_list_srv = None
        self._cm_load_srv = None
        self._cm_switch_srv = None
        self._streaming_loaded = False

        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.joint_traj_pub = rospy.Publisher(
            "/position_joint_trajectory_controller/command", 
            JointTrajectory, 
            queue_size=10
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        time.sleep(1)
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", FrankaState, self._set_currpos
        )

        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.joint_target_pub = rospy.Publisher(
            f"/{self.JOINT_STREAM_CONTROLLER}/joint_target",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )

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

    def _get_arm_id(self) -> str:
        for key in ("/franka_control/arm_id", "/franka_state_controller/arm_id", "/arm_id"):
            try:
                value = rospy.get_param(key)
                if isinstance(value, str) and value:
                    return value
            except Exception:
                pass
        return "panda"

    def _resolve_controller_manager_namespace(self, timeout_s: float = 8.0) -> str:
        if ListControllers is None:
            raise RuntimeError("controller_manager_msgs is not available in this environment")

        default_ns = "/controller_manager"
        try:
            rospy.wait_for_service(f"{default_ns}/list_controllers", timeout=timeout_s)
            return default_ns
        except Exception:
            pass

        if rosgraph is None:
            raise RuntimeError("rosgraph is not available; cannot discover controller_manager services")

        master = rosgraph.masterapi.Master(rospy.get_name())
        _, _, services = master.getSystemState()
        candidates = [name for name, _providers in services if name.endswith("/list_controllers")]
        if not candidates:
            raise RuntimeError("No controller_manager/list_controllers service found")

        best = sorted(candidates, key=len)[0]
        return best[: -len("/list_controllers")]

    def _ensure_controller_manager_clients(self, timeout_s: float = 8.0) -> None:
        if self._cm_list_srv is not None and self._cm_load_srv is not None and self._cm_switch_srv is not None:
            return
        if self._cm_ns is None:
            self._cm_ns = self._resolve_controller_manager_namespace(timeout_s=timeout_s)

        rospy.wait_for_service(f"{self._cm_ns}/list_controllers", timeout=timeout_s)
        rospy.wait_for_service(f"{self._cm_ns}/load_controller", timeout=timeout_s)
        rospy.wait_for_service(f"{self._cm_ns}/switch_controller", timeout=timeout_s)

        self._cm_list_srv = rospy.ServiceProxy(f"{self._cm_ns}/list_controllers", ListControllers)
        self._cm_load_srv = rospy.ServiceProxy(f"{self._cm_ns}/load_controller", LoadController)
        self._cm_switch_srv = rospy.ServiceProxy(f"{self._cm_ns}/switch_controller", SwitchController)

    def _list_controllers(self):
        self._ensure_controller_manager_clients()
        return self._cm_list_srv().controller

    def _is_controller_loaded(self, name: str) -> bool:
        try:
            for c in self._list_controllers():
                if c.name == name:
                    return True
        except Exception:
            pass
        return False

    def _load_controller(self, name: str) -> bool:
        self._ensure_controller_manager_clients()
        try:
            resp = self._cm_load_srv(name)
            return bool(getattr(resp, "ok", False))
        except Exception as e:
            rospy.logwarn(f"load_controller({name}) failed: {e}")
            return False

    def _switch_controllers(self, start, stop) -> bool:
        self._ensure_controller_manager_clients()
        strict = SwitchControllerRequest.STRICT if SwitchControllerRequest is not None else 2
        resp = self._cm_switch_srv(start, stop, strict, True, 0.0)
        return bool(getattr(resp, "ok", False))

    def _configure_streaming_joint_controller_params(self) -> None:
        arm_id = self._get_arm_id()
        self.joint_names = [f"{arm_id}_joint{i+1}" for i in range(7)]

        base = f"/{self.JOINT_STREAM_CONTROLLER}"
        rospy.set_param(f"{base}/type", "serl_franka_joint_streaming_controller/StreamingJointImpedanceController")
        rospy.set_param(f"{base}/arm_id", arm_id)
        rospy.set_param(f"{base}/joint_names", self.joint_names)
        rospy.set_param(f"{base}/target_topic", "joint_target")

        rospy.set_param(f"{base}/k_gains", [200.0, 200.0, 200.0, 200.0, 80.0, 50.0, 30.0])
        rospy.set_param(f"{base}/d_gains", [20.0, 20.0, 20.0, 20.0, 10.0, 8.0, 6.0])
        rospy.set_param(f"{base}/max_position_error", 0.15)
        rospy.set_param(f"{base}/command_timeout", 0.2)
        rospy.set_param(f"{base}/delta_tau_max", 1.0)

        rospy.set_param(f"{base}/estimate_dq_from_q", True)
        rospy.set_param(f"{base}/dq_max", 0.5)
        rospy.set_param(f"{base}/ddq_max", 2.0)
        rospy.set_param(f"{base}/q_filter", 0.01)
        rospy.set_param(f"{base}/dq_filter", 0.05)

    def _preload_streaming_joint_controller(self, timeout_s: float = 10.0) -> bool:
        if LoadController is None:
            rospy.logwarn("controller_manager_msgs is missing; joint streaming control is disabled")
            return False

        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                self._ensure_controller_manager_clients(timeout_s=2.0)
                self._configure_streaming_joint_controller_params()
                if not self._is_controller_loaded(self.JOINT_STREAM_CONTROLLER):
                    ok = self._load_controller(self.JOINT_STREAM_CONTROLLER)
                    if not ok:
                        last_error = "load_controller returned ok=false"
                        time.sleep(0.5)
                        continue
                self._streaming_loaded = True
                return True
            except Exception as e:
                last_error = str(e)
                time.sleep(0.5)
        rospy.logwarn(f"Failed to preload joint streaming controller: {last_error}")
        return False

    def start_joint_control(self) -> bool:
        with self._command_lock:
            if self._resetting.is_set():
                return False
            if not self._streaming_loaded:
                self._preload_streaming_joint_controller(timeout_s=10.0)
            if not self._streaming_loaded:
                return False

            ok = self._switch_controllers([self.JOINT_STREAM_CONTROLLER], [self.CARTESIAN_CONTROLLER])
            if ok:
                self.control_mode = "joint"
            return ok

    def stop_joint_control(self) -> bool:
        with self._command_lock:
            if self._resetting.is_set():
                return False
            ok = self._switch_controllers([self.CARTESIAN_CONTROLLER], [self.JOINT_STREAM_CONTROLLER])
            if ok:
                self.control_mode = "cartesian"
            return ok

    def send_joint_target_q(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=np.float64).reshape((7,))
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.joint_names)
        msg.position = [float(x) for x in q_target.tolist()]
        self.joint_target_pub.publish(msg)

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
            proc = getattr(self, "imp", None)
            if proc is not None and proc.poll() is None:
                return
            self.imp = subprocess.Popen(
                [
                    "roslaunch",
                    self.ros_pkg_name,
                    "impedance.launch",
                    "robot_ip:=" + self.robot_ip,
                    f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            time.sleep(3)
            self.control_mode = "cartesian"
            self._preload_streaming_joint_controller(timeout_s=10.0)

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
            self.control_mode = "stopped"

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
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
            if self.control_mode != "cartesian":
                raise RuntimeError(f"Cartesian pose control is not active (mode={self.control_mode})")
            assert len(pose) == 7
            msg = geom_msg.PoseStamped()
            msg.header.frame_id = "0"
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
            msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
            self.eepub.publish(msg)

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
            }

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        dq = np.array(list(msg.dq)).reshape((7,))
        q = np.array(list(msg.q)).reshape((7,))
        force = np.array(list(msg.K_F_ext_hat_K)[:3])
        torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        with self._state_lock:
            self.pos = pose
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
    gripper_lock = threading.RLock()

    def _busy(action: str):
        return jsonify(
            {
                "ok": False,
                "busy": True,
                "action": action,
                "resetting": robot_server.is_resetting(),
            }
        )

    def _get_gripper_position() -> float:
        if gripper_server is None:
            return 0.0
        with gripper_lock:
            pos = getattr(gripper_server, "gripper_pos", 0.0)
            try:
                return float(pos)
            except Exception:
                return 0.0


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
            try:
                robot_server.stop_joint_control()
            except Exception:
                pass
            robot_server.start_impedance()
            return jsonify({"ok": True})

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("stopimp")
            try:
                robot_server.stop_joint_control()
            except Exception:
                pass
            robot_server.stop_impedance()
            return jsonify({"ok": True})

    # Route for switching to joint streaming torque control
    @webapp.route("/start_joint_control", methods=["POST"])
    def start_joint_control():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("start_joint_control")
            robot_server.clear()
            ok = robot_server.start_joint_control()
            if not ok:
                return jsonify({"ok": False, "error": "failed to start joint control"}), 500
            return jsonify({"ok": True})

    # Route for switching back to Cartesian impedance control
    @webapp.route("/stop_joint_control", methods=["POST"])
    def stop_joint_control():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("stop_joint_control")
            robot_server.clear()
            ok = robot_server.stop_joint_control()
            if not ok:
                return jsonify({"ok": False, "error": "failed to stop joint control"}), 500
            return jsonify({"ok": True})

    # Route for streaming joint targets (q only)
    @webapp.route("/joint_q", methods=["POST"])
    def joint_q():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("joint_q")
            data = request.get_json(silent=True) or {}
            q = data.get("q", None)
            if q is None:
                q = data.get("arr", None)
            if q is None:
                return jsonify({"ok": False, "error": "missing 'q' (or 'arr')"}), 400
            robot_server.send_joint_target_q(np.array(q, dtype=np.float64))
            return jsonify({"ok": True})
    
    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        robot_server.wait_for_state(timeout_s=2.0)
        state = robot_server.get_state_copy()
        xyz = state["pose"][:3]
        r = R.from_quat(state["pose"][3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

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
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("open_gripper")
        with gripper_lock:
            gripper_server.open()
        return jsonify({"ok": True, "started": True})

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("activate_gripper")
        with gripper_lock:
            gripper_server.activate_gripper()
        return jsonify({"ok": True})

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("reset_gripper")
        with gripper_lock:
            gripper_server.reset_gripper()
        return jsonify({"ok": True})

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("open_gripper")
        with gripper_lock:
            gripper_server.open()
        return jsonify({"ok": True})

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("close_gripper")
        with gripper_lock:
            gripper_server.close()
        return jsonify({"ok": True})

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        print("close")
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("close_gripper_slow")
        with gripper_lock:
            gripper_server.close_slow()
        return jsonify({"ok": True})

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        if gripper_server is None:
            return jsonify({"ok": False, "error": "No gripper configured"}), 400
        if robot_server.is_resetting():
            return _busy("move_gripper")
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        with gripper_lock:
            gripper_server.move(pos)
        return jsonify({"ok": True})

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
            try:
                robot_server.move(pos)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 400
            return jsonify({"ok": True})

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
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        with robot_server._try_command() as acquired:
            if not acquired:
                return _busy("update_param")
            if robot_server.control_mode != "cartesian":
                return jsonify({"ok": False, "error": f"cartesian impedance not active (mode={robot_server.control_mode})"}), 400
            reconf_client.update_configuration(request.json)
            return jsonify({"ok": True})

    @webapp.route("/status", methods=["POST"])
    def status():
        return jsonify(
            {
                "ok": True,
                "resetting": robot_server.is_resetting(),
                "state_ready": robot_server._state_ready.is_set(),
                "mode": getattr(robot_server, "control_mode", "unknown"),
            }
        )
    
    @webapp.route("/reset_all", methods = ["POST"])
    def reset_all():
        data = request.get_json(silent=True) or {}
        max_s = float(data.get("max_s", 10.0))
        hz = float(data.get("hz", 30.0))
        pos_tol = float(data.get("pos_tol", 0.01))
        rot_tol_rad = float(data.get("rot_tol_rad", 0.2))
        goal =  np.array([0.5671124922989944,6.47218270568564e-05,0.4951717570264977,3.1406597193535584,-0.06601965456071524,4.5924120475993035e-05])
        goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        pos = np.array(goal).astype(np.float32)
        started = robot_server.start_hold_pose_async(
            pos, max_s=max_s, hz=hz, pos_tol=pos_tol, rot_tol_rad=rot_tol_rad
        )
        if not started:
            return _busy("reset_all")
        if gripper_server is not None:
            with gripper_lock:
                gripper_server.open()
        return jsonify({"ok": True, "started": True})

    webapp.run(host=FLAGS.flask_url, threaded=True)


if __name__ == "__main__":
    app.run(main)
