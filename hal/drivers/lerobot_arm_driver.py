"""HAL driver for LeRobot-based robot arms (SO-101, Koch v1.1, SO-100).

Wraps the upstream HuggingFace LeRobot Python SDK and exposes it through
the PhyAgentOS ``BaseDriver`` interface.  The driver communicates with the
physical arm over serial USB using Feetech STS3215 servos.

The ``solo-cli`` tool (``solo robo``) can be used alongside this driver
for motor setup, calibration, dataset recording, training, and inference.
This driver focuses on the real-time HAL control loop, while solo-cli
provides the higher-level ML pipeline.

Supported ``--driver`` names (registered in ``hal/drivers/__init__.py``):
    so101_follower, koch_follower

Typical invocation::

    python hal/hal_watchdog.py \\
        --driver so101_follower \\
        --driver-config examples/so101_follower.driver.json

Environment variables (fallbacks when config keys are absent)::

    LEROBOT_ARM_TYPE        so101_follower | koch_follower | so100_follower
    LEROBOT_ARM_PORT        /dev/ttyACM0  (serial USB device)
    LEROBOT_ROBOT_ID        my_so101_arm
    LEROBOT_LEADER_PORT     /dev/ttyACM1  (optional, for teleoperation)
    LEROBOT_LEADER_TYPE     so101_leader | koch_leader
    LEROBOT_LOOP_HZ         50.0
    LEROBOT_MAX_STEP_DEG    20.0
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hal.base_driver import BaseDriver

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

# ── Joint topology ─────────────────────────────────────────────────────
# SO-101, Koch v1.1, and SO-100 all share the same 6-DOF layout.
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Map from arm_type → (module_path, robot_class_name, config_class_name)
_LEROBOT_ARM_IMPORTS: dict[str, tuple[str, str, str]] = {
    "so101_follower": (
        "lerobot.robots.so101_follower",
        "SO101Follower",
        "SO101FollowerConfig",
    ),
    "so101": (
        "lerobot.robots.so101_follower",
        "SO101Follower",
        "SO101FollowerConfig",
    ),
    "koch_follower": (
        "lerobot.robots.koch_follower",
        "KochFollower",
        "KochFollowerConfig",
    ),
    "koch": (
        "lerobot.robots.koch_follower",
        "KochFollower",
        "KochFollowerConfig",
    ),
    "so100_follower": (
        "lerobot.robots.so100_follower",
        "SO100Follower",
        "SO100FollowerConfig",
    ),
    "so100": (
        "lerobot.robots.so100_follower",
        "SO100Follower",
        "SO100FollowerConfig",
    ),
}


def _parse_float(raw: str | None, default: float) -> float:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class LeRobotArmDriver(BaseDriver):
    """Control a LeRobot-based arm through the HuggingFace LeRobot SDK.

    Two levels of control are exposed:

    1. **Direct joint control** — ``set_joint_targets``, ``set_gripper``,
       ``get_joint_positions``, ``go_to_rest`` use the LeRobot
       ``robot.send_action()`` / ``robot.get_observation()`` API.

    2. **Solo-CLI integration** — ``solo_calibrate``, ``solo_teleop``,
       ``solo_record``, ``solo_train``, ``solo_inference`` shell out to
       ``solo robo`` for higher-level ML pipeline operations.
    """

    _default_arm_type: str = "so101_follower"
    _default_profile: str = "so101_lekoch.md"

    def __init__(
        self,
        gui: bool = False,
        *,
        arm_type: str | None = None,
        port: str | None = None,
        robot_id: str | None = None,
        leader_port: str | None = None,
        leader_type: str | None = None,
        leader_id: str | None = None,
        loop_hz: float | None = None,
        use_degrees: bool = True,
        cameras: dict[str, Any] | None = None,
        max_step_deg: float | None = None,
        reconnect_policy: str = "auto",
        **_kwargs: Any,
    ) -> None:
        self._gui = gui

        # ── Follower arm (the robot) ──
        self.arm_type = (
            arm_type or os.environ.get("LEROBOT_ARM_TYPE") or self._default_arm_type
        ).strip().lower()
        self.port = (port or os.environ.get("LEROBOT_ARM_PORT", "")).strip()
        self.robot_id = (
            robot_id or os.environ.get("LEROBOT_ROBOT_ID") or f"my_{self.arm_type}"
        ).strip()

        # ── Leader arm (for teleoperation) — optional ──
        self.leader_port = (
            leader_port or os.environ.get("LEROBOT_LEADER_PORT", "")
        ).strip()
        self.leader_type = (
            leader_type or os.environ.get("LEROBOT_LEADER_TYPE", "")
        ).strip().lower()
        self.leader_id = (
            leader_id or os.environ.get("LEROBOT_LEADER_ID", "")
        ).strip()

        # ── Control parameters ──
        self.loop_hz = max(
            loop_hz if loop_hz is not None
            else _parse_float(os.environ.get("LEROBOT_LOOP_HZ"), 50.0),
            1.0,
        )
        self.use_degrees = use_degrees
        self.cameras_cfg = cameras or {}
        self.max_step_deg = (
            max_step_deg if max_step_deg is not None
            else _parse_float(os.environ.get("LEROBOT_MAX_STEP_DEG"), 20.0)
        )
        self.reconnect_policy = reconnect_policy

        # ── Internal state ──
        self._objects: dict[str, dict] = {}
        self._robot: Any | None = None
        self._runtime_state: dict[str, Any] = {
            "robots": {self.robot_id: self._make_robot_state()},
        }

    # ═══════════════════════════════════════════════════════════════════
    #  BaseDriver required methods
    # ═══════════════════════════════════════════════════════════════════

    def get_profile_path(self) -> Path:
        return _PROFILES_DIR / self._default_profile

    def load_scene(self, scene: dict[str, dict]) -> None:
        self._objects = dict(scene)

    def get_scene(self) -> dict[str, dict]:
        return dict(self._objects)

    def execute_action(self, action_type: str, params: dict) -> str:
        try:
            self._validate_robot_id(params)

            # ── Connection lifecycle ──
            if action_type == "connect_robot":
                return "Robot connected." if self.connect() else self._conn_error()
            if action_type == "check_connection":
                return "connected" if self.health_check() else "disconnected"
            if action_type == "disconnect_robot":
                self.disconnect()
                return "Robot disconnected."

            # ── Direct joint control ──
            if action_type == "set_joint_targets":
                return self._execute_set_joint_targets(params)
            if action_type == "set_gripper":
                return self._execute_set_gripper(params)
            if action_type == "get_joint_positions":
                return self._execute_get_joint_positions()
            if action_type == "go_to_rest":
                return self._execute_go_to_rest()

            # ── Solo-CLI pipeline commands ──
            if action_type == "solo_calibrate":
                return self._execute_solo_command("--calibrate", params)
            if action_type == "solo_teleop":
                return self._execute_solo_command("--teleop", params)
            if action_type == "solo_record":
                return self._execute_solo_command("--record", params)
            if action_type == "solo_train":
                return self._execute_solo_command("--train", params)
            if action_type == "solo_inference":
                return self._execute_solo_command("--inference", params)
            if action_type == "solo_replay":
                return self._execute_solo_command("--replay", params)

            # ── LeRobot native CLI commands ──
            if action_type == "lerobot_calibrate":
                return self._execute_lerobot_calibrate(params)
            if action_type == "lerobot_teleop":
                return self._execute_lerobot_teleop(params)

            return f"Unknown action: {action_type}"
        except ValueError as exc:
            return self._error_result(str(exc))
        except Exception as exc:
            return self._error_result(f"{action_type} failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════
    #  Connection lifecycle
    # ═══════════════════════════════════════════════════════════════════

    def connect(self) -> bool:
        if self.is_connected():
            self._set_connection_status("connected", last_error=None)
            self._touch_heartbeat()
            return True

        if not self.port:
            self._set_connection_status("error", last_error="port not configured")
            return False

        try:
            robot_cls, config_cls = self._import_robot_classes(self.arm_type)
            config = config_cls(
                port=self.port,
                id=self.robot_id,
                use_degrees=self.use_degrees,
            )
            self._robot = robot_cls(config)
            self._robot.connect()

            # Validate connection by reading one observation.
            obs = self._robot.get_observation()
            self._update_joint_state(obs)

            self._set_connection_status("connected", last_error=None)
            self._touch_heartbeat()
            self._set_nav_state(mode="idle", status="idle", last_error=None)
            return True
        except Exception as exc:
            self._robot = None
            self._set_connection_status("error", last_error=str(exc))
            return False

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
            self._robot = None
        self._set_connection_status("disconnected", last_error=None)
        self._set_nav_state(mode="idle", status="idle", last_error=None)

    def is_connected(self) -> bool:
        return self._robot is not None and bool(
            getattr(self._robot, "is_connected", False)
        )

    def health_check(self) -> bool:
        if not self.is_connected():
            if self.reconnect_policy == "auto":
                self._inc_reconnect_attempts()
                self._set_connection_status("reconnecting", last_error="disconnected")
                connected = self.connect()
                if not connected:
                    self._set_nav_state(
                        mode="idle", status="failed",
                        last_error=self._robot_state()
                        .get("connection_state", {})
                        .get("last_error"),
                    )
                return connected
            self._set_connection_status("disconnected", last_error="disconnected")
            return False

        try:
            obs = self._robot.get_observation()
            self._update_joint_state(obs)
            self._touch_heartbeat()
            self._set_connection_status("connected", last_error=None)
            return True
        except Exception as exc:
            self._set_connection_status("error", last_error=str(exc))
            return False

    def get_runtime_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._runtime_state)

    def close(self) -> None:
        self.disconnect()

    # ═══════════════════════════════════════════════════════════════════
    #  Direct joint-control actions
    # ═══════════════════════════════════════════════════════════════════

    def _execute_set_joint_targets(self, params: dict[str, Any]) -> str:
        if not self.is_connected():
            return self._conn_error()

        joints = params.get("joints")
        if not isinstance(joints, dict) or not joints:
            return "Error: joints must be a non-empty dict mapping joint names to values."

        try:
            import torch

            obs = self._robot.get_observation()
            state_key = self._find_state_key(obs)
            if state_key and state_key in obs:
                action = obs[state_key].clone().float()
            else:
                action = torch.zeros(len(JOINT_NAMES), dtype=torch.float32)

            for name, value in joints.items():
                if name in JOINT_NAMES:
                    idx = JOINT_NAMES.index(name)
                    action[idx] = float(value)

            self._robot.send_action(action)
            self._touch_heartbeat()
            return f"Applied joint targets: {joints}"
        except Exception as exc:
            return self._error_result(f"set_joint_targets failed: {exc}")

    def _execute_set_gripper(self, params: dict[str, Any]) -> str:
        if not self.is_connected():
            return self._conn_error()
        if "value" not in params:
            return "Error: 'value' parameter is required."

        try:
            import torch

            value = float(params["value"])
            obs = self._robot.get_observation()
            state_key = self._find_state_key(obs)
            if state_key and state_key in obs:
                action = obs[state_key].clone().float()
            else:
                action = torch.zeros(len(JOINT_NAMES), dtype=torch.float32)

            action[-1] = value  # gripper is always the last joint
            self._robot.send_action(action)
            self._touch_heartbeat()
            return f"Set gripper to {value:.2f}."
        except Exception as exc:
            return self._error_result(f"set_gripper failed: {exc}")

    def _execute_get_joint_positions(self) -> str:
        if not self.is_connected():
            return self._conn_error()
        try:
            obs = self._robot.get_observation()
            self._update_joint_state(obs)
            self._touch_heartbeat()

            state_key = self._find_state_key(obs)
            if not state_key or state_key not in obs:
                return "Error: no joint state observation available."

            state = obs[state_key]
            positions = {
                name: round(float(state[i]), 2)
                for i, name in enumerate(JOINT_NAMES)
                if i < len(state)
            }
            return f"Joint positions: {json.dumps(positions)}"
        except Exception as exc:
            return self._error_result(f"get_joint_positions failed: {exc}")

    def _execute_go_to_rest(self) -> str:
        if not self.is_connected():
            return self._conn_error()
        try:
            import torch

            rest = torch.zeros(len(JOINT_NAMES), dtype=torch.float32)
            self._robot.send_action(rest)
            self._touch_heartbeat()
            return "Robot moved to rest position."
        except Exception as exc:
            return self._error_result(f"go_to_rest failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════
    #  Solo-CLI integration
    # ═══════════════════════════════════════════════════════════════════

    def _execute_solo_command(self, flag: str, params: dict[str, Any]) -> str:
        cmd = ["solo", "robo", flag]
        if params.get("auto_confirm", True):
            cmd.append("--yes")

        timeout = int(params.get("timeout_s", 300))
        self._set_nav_state(
            mode="solo_cli", status="running",
            goal={"command": " ".join(cmd)}, last_error=None,
        )

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(Path.home()),
            )
            output = (result.stdout or "").strip()
            if result.returncode != 0:
                err = (result.stderr or "").strip()
                self._set_nav_state(
                    mode="idle", status="failed",
                    last_error=f"solo exited {result.returncode}: {err}",
                )
                return f"Error: solo robo {flag} failed (rc={result.returncode}): {err}"

            self._set_nav_state(mode="idle", status="stopped", last_error=None)
            return f"solo robo {flag} completed. Output: {output[:500]}"
        except subprocess.TimeoutExpired:
            self._set_nav_state(
                mode="idle", status="failed",
                last_error=f"solo robo {flag} timed out after {timeout}s",
            )
            return f"Error: solo robo {flag} timed out after {timeout}s."
        except FileNotFoundError:
            self._set_nav_state(
                mode="idle", status="failed", last_error="solo-cli not installed",
            )
            return (
                "Error: solo-cli is not installed. "
                "Install with: pip install solo-cli"
            )

    # ═══════════════════════════════════════════════════════════════════
    #  LeRobot native CLI integration
    # ═══════════════════════════════════════════════════════════════════

    def _execute_lerobot_calibrate(self, params: dict[str, Any]) -> str:
        target = params.get("target", "follower")
        if target == "leader":
            if not self.leader_port or not self.leader_type:
                return "Error: leader_port and leader_type required."
            cmd = [
                sys.executable, "-m", "lerobot.calibrate",
                f"--teleop.type={self.leader_type}",
                f"--teleop.port={self.leader_port}",
                f"--teleop.id={self.leader_id or 'leader'}",
            ]
        else:
            if not self.port:
                return "Error: port must be configured."
            cmd = [
                sys.executable, "-m", "lerobot.calibrate",
                f"--robot.type={self.arm_type}",
                f"--robot.port={self.port}",
                f"--robot.id={self.robot_id}",
            ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=int(params.get("timeout_s", 120)),
            )
            if result.returncode != 0:
                return f"Error: calibration failed: {(result.stderr or '').strip()}"
            return f"Calibration for {target} completed successfully."
        except Exception as exc:
            return f"Error: calibration failed: {exc}"

    def _execute_lerobot_teleop(self, params: dict[str, Any]) -> str:
        if not self.port or not self.leader_port:
            return "Error: both port and leader_port must be configured."

        leader_type = self.leader_type or self.arm_type.replace("_follower", "_leader")
        cmd = [
            sys.executable, "-m", "lerobot.teleoperate",
            f"--robot.type={self.arm_type}",
            f"--robot.port={self.port}",
            f"--robot.id={self.robot_id}",
            f"--teleop.type={leader_type}",
            f"--teleop.port={self.leader_port}",
            f"--teleop.id={self.leader_id or 'leader'}",
        ]

        timeout = int(params.get("timeout_s", 600))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            return f"Teleoperation ended (rc={result.returncode})."
        except subprocess.TimeoutExpired:
            return f"Teleoperation timed out after {timeout}s."
        except Exception as exc:
            return f"Error: teleoperation failed: {exc}"

    # ═══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _import_robot_classes(arm_type: str) -> tuple[type, type]:
        import importlib

        spec = _LEROBOT_ARM_IMPORTS.get(arm_type)
        if spec is None:
            available = ", ".join(sorted(_LEROBOT_ARM_IMPORTS))
            raise ValueError(f"Unknown arm_type {arm_type!r}. Available: {available}")

        module_path, robot_cls_name, config_cls_name = spec
        module = importlib.import_module(module_path)
        return getattr(module, robot_cls_name), getattr(module, config_cls_name)

    @staticmethod
    def _find_state_key(obs: dict[str, Any]) -> str | None:
        for key in ("observation.state", "state", "position"):
            if key in obs:
                return key
        for key, value in obs.items():
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                return key
        return None

    def _update_joint_state(self, obs: dict[str, Any]) -> None:
        state_key = self._find_state_key(obs)
        if not state_key or state_key not in obs:
            return
        state_tensor = obs[state_key]
        positions = {}
        for i, name in enumerate(JOINT_NAMES):
            if i < len(state_tensor):
                positions[name] = round(float(state_tensor[i]), 2)
        rstate = self._robot_state()
        pose = dict(rstate.get("robot_pose", {}))
        pose["joints"] = positions
        pose["stamp"] = self._stamp()
        rstate["robot_pose"] = pose

    def _validate_robot_id(self, params: dict[str, Any]) -> None:
        requested = str(params.get("robot_id", "")).strip()
        if requested and requested != self.robot_id:
            raise ValueError(
                f"robot_id mismatch: requested={requested}, configured={self.robot_id}"
            )

    def _conn_error(self) -> str:
        details = self._robot_state().get("connection_state", {}).get("last_error")
        suffix = f" Details: {details}" if details else ""
        return (
            "Connection error: robot is not connected. "
            "Run connect_robot first and ensure the arm is powered and USB-connected."
            + suffix
        )

    def _error_result(self, reason: str) -> str:
        self._set_nav_state(mode="idle", status="failed", last_error=reason)
        return f"Error: {reason}"

    # ── Runtime state management ───────────────────────────────────────

    def _robot_state(self) -> dict[str, Any]:
        robots = self._runtime_state.setdefault("robots", {})
        if self.robot_id not in robots:
            robots[self.robot_id] = self._make_robot_state()
        return robots[self.robot_id]

    def _make_robot_state(self) -> dict[str, Any]:
        stamp = self._stamp()
        return {
            "connection_state": {
                "status": "disconnected",
                "transport": "serial_usb",
                "host": self.port or "unknown",
                "arm_type": self.arm_type,
                "last_heartbeat": None,
                "last_error": None,
                "reconnect_attempts": 0,
            },
            "robot_pose": {
                "frame": "base",
                "joints": {name: 0.0 for name in JOINT_NAMES},
                "stamp": stamp,
            },
            "nav_state": {
                "mode": "idle",
                "status": "idle",
                "goal": None,
                "last_error": None,
            },
        }

    def _set_connection_status(self, status: str, last_error: str | None) -> None:
        state = self._robot_state()
        conn = dict(state.get("connection_state", {}))
        conn.update({
            "status": status,
            "transport": "serial_usb",
            "host": self.port or "unknown",
            "arm_type": self.arm_type,
            "last_error": last_error,
        })
        state["connection_state"] = conn

    def _set_nav_state(
        self, *, mode: str, status: str,
        goal: dict[str, Any] | None = None,
        last_error: str | None = None,
    ) -> None:
        state = self._robot_state()
        state["nav_state"] = {
            "mode": mode, "status": status,
            "goal": goal, "last_error": last_error,
        }

    def _inc_reconnect_attempts(self) -> None:
        state = self._robot_state()
        conn = dict(state.get("connection_state", {}))
        conn["reconnect_attempts"] = int(conn.get("reconnect_attempts", 0)) + 1
        state["connection_state"] = conn

    def _touch_heartbeat(self) -> None:
        state = self._robot_state()
        conn = dict(state.get("connection_state", {}))
        conn["last_heartbeat"] = self._stamp()
        state["connection_state"] = conn

    @staticmethod
    def _stamp() -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )


# ═══════════════════════════════════════════════════════════════════════
#  Concrete aliases for the driver registry
# ═══════════════════════════════════════════════════════════════════════


class SO101FollowerDriver(LeRobotArmDriver):
    """SO-101 follower arm driver."""
    _default_arm_type = "so101_follower"
    _default_profile = "so101_lekoch.md"


class KochFollowerDriver(LeRobotArmDriver):
    """Koch v1.1 follower arm driver."""
    _default_arm_type = "koch_follower"
    _default_profile = "so101_lekoch.md"
