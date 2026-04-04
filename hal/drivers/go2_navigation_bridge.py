from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pexpect
from PIL import Image

from hal.navigation.bridge import ActionCommand, LocalHorizonCommand, RobotBridge
from hal.navigation.models import Observation


VIDEO_PORT = 5220
STATE_PORT = 5222
OCCUPANCY_PORT = 5223
DEPTH_PORT = 5224
MOTION_PORT = 8000


@dataclass
class Go2BridgeConfig:
    host_bind: str = "0.0.0.0"
    video_port: int = VIDEO_PORT
    state_port: int = STATE_PORT
    occupancy_port: int = OCCUPANCY_PORT
    depth_port: int = DEPTH_PORT
    motion_port: int = MOTION_PORT
    ssh_host: str = "192.168.86.26"
    ssh_user: str = "unitree"
    ssh_password: str = ""
    ssh_options: tuple[str, ...] = ("-o", "StrictHostKeyChecking=accept-new")
    remote_project_dir: str = "/home/unitree/OpenEmbodiedAgent"
    remote_python: str = "python3"
    remote_setup: str = ""
    remote_ros_choice: str = "1"
    remote_sudo_password: str = ""
    auto_start_remote: bool = False
    remote_livox_setup: str = ""
    remote_livox_launch: str = ""
    remote_data_command: str = ""
    remote_motion_command: str = ""
    remote_data_video_backend: str = "auto"
    remote_data_video_index: int = 4
    remote_motion_backend: str = "auto"
    remote_motion_sdk_python_path: str = "/home/unitree/unitree_sdk2_python"
    remote_motion_network_interface: str = ""
    remote_motion_require_subscriber: bool = True
    remote_sync_before_start: bool = False
    remote_sync_paths: tuple[str, ...] = ("move",)
    remote_sync_excludes: tuple[str, ...] = ("__pycache__", "*.pyc", ".pytest_cache", ".venv")
    remote_startup_delay_s: float = 2.0
    remote_observation_wait_timeout_s: float = 20.0
    forward_speed_x: float = 0.55
    turn_speed_z: float = 0.95
    motion_confirm_timeout_s: float = 8.0
    motion_confirm_translation_m: float = 0.05
    motion_confirm_rotation_deg: float = 8.0
    horizon_confirm_timeout_s: float = 1.5
    horizon_confirm_translation_m: float = 0.02
    horizon_confirm_rotation_deg: float = 3.0
    lateral_speed_y: float = 0.32


class VideoStateOccupancyReceiver:
    def __init__(self, config: Go2BridgeConfig):
        self.config = config
        self.running = False
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.latest_occupancy: np.ndarray | None = None
        self.latest_timestamp = 0.0
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.threads = [
            threading.Thread(target=self._receive_video, daemon=True),
            threading.Thread(target=self._receive_state, daemon=True),
            threading.Thread(target=self._receive_occupancy, daemon=True),
            threading.Thread(target=self._receive_depth, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.running = False

    def get_latest(self) -> Observation:
        with self.lock:
            return Observation(
                rgb=None if self.latest_rgb is None else self.latest_rgb.copy(),
                depth_m=None if self.latest_depth is None else self.latest_depth.copy(),
                occupancy=None if self.latest_occupancy is None else self.latest_occupancy.copy(),
                pose_xy_yaw=self.latest_pose,
                timestamp=self.latest_timestamp,
            )

    def _receive_video(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.config.host_bind, self.config.video_port))
        sock.settimeout(1.0)
        buffers: dict[int, dict[int, bytes]] = defaultdict(dict)
        totals: dict[int, int] = {}
        while self.running:
            try:
                packet, _ = sock.recvfrom(65536)
            except (socket.timeout, OSError):
                continue
            if len(packet) < 10:
                continue
            frame_id, total, idx, payload_len = struct.unpack("!IHHH", packet[:10])
            payload = packet[10:]
            if len(payload) != payload_len:
                continue
            buffers[frame_id][idx] = payload
            totals[frame_id] = total
            if len(buffers[frame_id]) == total:
                data = b"".join(buffers[frame_id][i] for i in range(total))
                try:
                    img = np.array(Image.open(__import__("io").BytesIO(data)).convert("RGB"))
                except Exception:
                    buffers.pop(frame_id, None)
                    totals.pop(frame_id, None)
                    continue
                with self.lock:
                    self.latest_rgb = img
                    self.latest_timestamp = time.time()
                buffers.pop(frame_id, None)
                totals.pop(frame_id, None)
        sock.close()

    def _receive_state(self) -> None:
        self._receive_stream(self.config.state_port, self._handle_state_payload)

    def _receive_occupancy(self) -> None:
        self._receive_stream(self.config.occupancy_port, self._handle_occupancy_payload)

    def _receive_depth(self) -> None:
        self._receive_stream(self.config.depth_port, self._handle_depth_payload)

    def _receive_stream(self, port: int, handler) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.config.host_bind, port))
        server.listen(1)
        server.settimeout(1.0)
        while self.running:
            try:
                conn, _ = server.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                conn.settimeout(1.0)
                while self.running:
                    try:
                        header = self._recv_exact(conn, 4)
                        payload_len = struct.unpack("!I", header)[0]
                        if payload_len == 0:
                            continue
                        payload = self._recv_exact(conn, payload_len)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    try:
                        handler(payload)
                    except Exception:
                        continue
        server.close()

    def _handle_state_payload(self, payload: bytes) -> None:
        if len(payload) != struct.calcsize("!ddd"):
            return
        x, y, yaw = struct.unpack("!ddd", payload)
        with self.lock:
            self.latest_pose = (float(x), float(y), float(yaw))
            self.latest_timestamp = time.time()

    def _handle_occupancy_payload(self, payload: bytes) -> None:
        if len(payload) < 8:
            return
        rows, cols = struct.unpack("!II", payload[:8])
        if rows == 0 or cols == 0 or len(payload[8:]) != rows * cols:
            return
        grid = np.frombuffer(payload[8:], dtype=np.uint8).reshape(rows, cols)
        with self.lock:
            self.latest_occupancy = grid.copy()
            self.latest_timestamp = time.time()

    def _handle_depth_payload(self, payload: bytes) -> None:
        if len(payload) < 8:
            return
        rows, cols = struct.unpack("!II", payload[:8])
        try:
            image = Image.open(__import__("io").BytesIO(payload[8:]))
        except Exception:
            return
        depth_mm = np.array(image, dtype=np.uint16)
        if depth_mm.shape[:2] != (rows, cols):
            return
        depth_m = depth_mm.astype(np.float32) * 0.001
        depth_m[depth_mm == 0] = np.nan
        with self.lock:
            self.latest_depth = depth_m
            self.latest_timestamp = time.time()

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            chunk = conn.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("connection closed")
            buf.extend(chunk)
        return bytes(buf)


class MotionCommandServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.running = False
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.websocket = None
        self.websocket_ready = threading.Event()
        self.initialized_ready = threading.Event()
        self.command_lock = threading.Lock()
        self.status_lock = threading.Condition()
        self.status_seq = 0
        self.last_status: dict[str, Any] | None = None
        self.next_command_id = 1
        self.start_error: str | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.initialized_ready.clear()
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)

    def is_connected(self) -> bool:
        return self.websocket is not None and self.websocket_ready.is_set()

    def send_atomic(self, action: ActionCommand, forward_speed_x: float, turn_speed_z: float) -> dict[str, Any]:
        if action.kind == "stop":
            return self._send_payload(1003, {})
        if action.kind == "forward":
            return self._send_payload(1008, {"controller": "closed_loop", "x": forward_speed_x, "y": 0.0, "z": -0.01, "target_distance_m": float(action.value)}, binary=[127])
        if action.kind == "turn_left":
            return self._send_payload(1008, {"controller": "closed_loop", "x": 0.0, "y": 0.0, "z": abs(turn_speed_z), "target_yaw_deg": float(action.value)}, binary=[127])
        if action.kind == "turn_right":
            return self._send_payload(1008, {"controller": "closed_loop", "x": 0.0, "y": 0.0, "z": -abs(turn_speed_z), "target_yaw_deg": float(action.value)}, binary=[127])
        return {"ok": False, "reason": f"unsupported action {action.kind}"}

    def send_horizon(self, command: LocalHorizonCommand, forward_speed_x: float, side_speed_y: float, turn_speed_z: float) -> dict[str, Any]:
        return self._send_payload(
            1008,
            {
                "controller": "local_horizon",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "goal_pose_body": {
                    "forward_m": float(command.forward_m),
                    "lateral_m": float(command.lateral_m),
                    "heading_rad": float(command.heading_rad),
                },
                "expires_in_s": float(command.valid_for_s),
                "sequence_id": int(command.sequence_id),
                "issued_at": float(command.issued_at),
                "source": command.source,
                "lookahead_xy": None if command.lookahead_xy is None else [float(command.lookahead_xy[0]), float(command.lookahead_xy[1])],
                "max_speeds": {"x": float(forward_speed_x), "y": float(side_speed_y), "z": float(turn_speed_z)},
            },
            binary=[127],
        )

    def _send_payload(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> dict[str, Any]:
        with self.command_lock:
            not_ready = self._wait_until_ready()
            if not_ready is not None:
                return not_ready
            command_id = self.next_command_id
            self.next_command_id += 1
            payload = {"command_id": command_id, **parameter}
            self._send_now(api_id, payload, binary=binary)
            return {"ok": True, "command_id": command_id, "parameter": payload}

    def _wait_until_ready(self) -> dict[str, Any] | None:
        if self.start_error is not None:
            return {"ok": False, "reason": self.start_error}
        if not self.websocket_ready.wait(timeout=5):
            return {"ok": False, "reason": "robot motion websocket not connected"}
        if not self.initialized_ready.wait(timeout=8):
            return {"ok": False, "reason": "robot motion websocket connected but initialization not finished"}
        return None

    def _run(self) -> None:
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
        except ImportError as exc:
            self.start_error = f"websockets import failed: {exc}"
            return

        async def handler(websocket):
            self.websocket = websocket
            self.websocket_ready.set()
            self.initialized_ready.set()
            try:
                async for raw_message in websocket:
                    self._record_robot_status(raw_message)
            except ConnectionClosed:
                pass
            finally:
                self.websocket = None
                self.websocket_ready.clear()
                self.initialized_ready.clear()

        async def main() -> None:
            async with websockets.serve(handler, self.host, self.port):
                while self.running:
                    await asyncio.sleep(0.1)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(main())
        except RuntimeError:
            pass

    def _send_now(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> None:
        if self.loop is None:
            raise RuntimeError("motion server loop is not running")
        future = asyncio.run_coroutine_threadsafe(self._send_async(api_id, parameter, binary=binary), self.loop)
        future.result(timeout=5)

    async def _send_async(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> None:
        if self.websocket is None:
            raise RuntimeError("robot motion websocket is not connected")
        payload: dict[str, Any] = {"api_id": api_id, "parameter": parameter}
        if binary is not None:
            payload["binary"] = binary
        await self.websocket.send(json.dumps(payload))

    def _record_robot_status(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        with self.status_lock:
            self.status_seq += 1
            self.last_status = payload
            self.status_lock.notify_all()


class Go2MoveBridge(RobotBridge):
    def __init__(self, config: Go2BridgeConfig | None = None) -> None:
        self.config = config or Go2BridgeConfig()
        self.receiver = VideoStateOccupancyReceiver(self.config)
        self.receiver.start()
        self.motion_server = MotionCommandServer(self.config.host_bind, self.config.motion_port)
        self.motion_server.start()
        self._remote_processes: list[pexpect.spawn] = []

    def describe_navigation_capabilities(self) -> dict[str, Any]:
        return {
            "has_rgb": True,
            "has_depth": True,
            "has_occupancy": True,
            "supports_local_horizon": True,
            "supports_obstacle_avoidance": True,
            "supports_external_map_assist": True,
        }

    def describe(self) -> dict[str, Any]:
        latest = self.receiver.get_latest()
        return {
            "host_bind": self.config.host_bind,
            "video_port": self.config.video_port,
            "state_port": self.config.state_port,
            "occupancy_port": self.config.occupancy_port,
            "depth_port": self.config.depth_port,
            "motion_port": self.config.motion_port,
            "ssh_host": self.config.ssh_host,
            "ssh_user": self.config.ssh_user,
            "remote_project_dir": self.config.remote_project_dir,
            "motion_connected": self.motion_server.is_connected(),
            "robot_motion_status": self.motion_server.last_status,
            "latest_observation_timestamp": latest.timestamp,
        }

    def stop_remote_services(self) -> dict[str, Any]:
        stopped = []
        for proc in self._remote_processes:
            try:
                if proc.isalive():
                    proc.sendintr()
                    time.sleep(0.5)
                if proc.isalive():
                    proc.close(force=True)
            finally:
                stopped.append(proc.pid)
        self._remote_processes = []
        return {"ok": True, "stopped_pids": stopped}

    def get_observation(self) -> Observation:
        obs = self.receiver.get_latest()
        if obs.timestamp <= 0.0:
            raise RuntimeError("no live RGB/state observation received from robot yet")
        return obs

    def get_motion_feedback(self) -> dict[str, Any] | None:
        return self.motion_server.last_status

    def execute(self, command: ActionCommand | LocalHorizonCommand) -> dict[str, Any]:
        before = self.receiver.get_latest()
        if isinstance(command, LocalHorizonCommand):
            send_result = self.motion_server.send_horizon(command, self.config.forward_speed_x, self.config.lateral_speed_y, self.config.turn_speed_z)
        else:
            send_result = self.motion_server.send_atomic(command, self.config.forward_speed_x, self.config.turn_speed_z)
        if not send_result.get("ok"):
            return send_result
        return {**send_result, **self._confirm_motion(command, before)}

    def wait_until_ready(self, timeout_s: float | None = None) -> dict[str, Any]:
        deadline = time.time() + (
            self.config.remote_observation_wait_timeout_s if timeout_s is None else float(timeout_s)
        )
        while time.time() < deadline:
            motion_connected = self.motion_server.is_connected()
            latest = self.receiver.get_latest()
            if motion_connected and latest.timestamp > 0.0:
                return {"ok": True, "motion_connected": True, "observation_ready": True}
            time.sleep(0.2)
        latest = self.receiver.get_latest()
        return {
            "ok": False,
            "motion_connected": self.motion_server.is_connected(),
            "observation_ready": latest.timestamp > 0.0,
            "reason": "bridge_not_ready",
        }

    def _confirm_motion(self, command: ActionCommand | LocalHorizonCommand, before: Observation) -> dict[str, Any]:
        if isinstance(command, ActionCommand) and command.kind == "stop":
            return {"motion_confirmed": True, "motion_reason": "stop_not_verified"}
        deadline = time.time() + (self.config.horizon_confirm_timeout_s if isinstance(command, LocalHorizonCommand) else self.config.motion_confirm_timeout_s)
        while time.time() < deadline:
            current = self.receiver.get_latest()
            if current.timestamp <= before.timestamp:
                time.sleep(0.2)
                continue
            delta_x = current.pose_xy_yaw[0] - before.pose_xy_yaw[0]
            delta_y = current.pose_xy_yaw[1] - before.pose_xy_yaw[1]
            delta_yaw = self._angle_diff(current.pose_xy_yaw[2], before.pose_xy_yaw[2])
            translation = math.hypot(delta_x, delta_y)
            rotation_deg = abs(math.degrees(delta_yaw))
            if isinstance(command, LocalHorizonCommand) and (translation >= self.config.horizon_confirm_translation_m or rotation_deg >= self.config.horizon_confirm_rotation_deg):
                return {"motion_confirmed": True, "motion_reason": "horizon_progress_observed", "odometry_delta_xy": [delta_x, delta_y], "odometry_delta_yaw_rad": delta_yaw}
            if isinstance(command, ActionCommand) and command.kind == "forward" and translation >= self.config.motion_confirm_translation_m:
                return {"motion_confirmed": True, "motion_reason": "odometry_translation_observed", "odometry_delta_xy": [delta_x, delta_y], "odometry_delta_yaw_rad": delta_yaw}
            if isinstance(command, ActionCommand) and command.kind in {"turn_left", "turn_right"} and rotation_deg >= self.config.motion_confirm_rotation_deg:
                return {"motion_confirmed": True, "motion_reason": "odometry_rotation_observed", "odometry_delta_xy": [delta_x, delta_y], "odometry_delta_yaw_rad": delta_yaw}
            time.sleep(0.2)
        return {"ok": not isinstance(command, ActionCommand), "motion_confirmed": False, "motion_reason": "odometry_change_not_observed"}

    @staticmethod
    def _angle_diff(current: float, previous: float) -> float:
        delta = current - previous
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        return delta
