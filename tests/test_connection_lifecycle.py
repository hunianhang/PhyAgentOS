from __future__ import annotations

import json
from pathlib import Path

from hal.drivers import load_driver
from hal.hal_watchdog import _poll_once
from hal.simulation.scene_io import load_environment_doc, save_environment_doc

_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"


def _write_action(workspace: Path, payload: dict) -> Path:
    action_file = workspace / "ACTION.md"
    action_file.write_text(
        f"{_FENCE_OPEN}\n{json.dumps(payload, indent=2)}\n{_FENCE_CLOSE}\n",
        encoding="utf-8",
    )
    return action_file


def _base_environment() -> dict:
    return {
        "schema_version": "oea.environment.v1",
        "scene_graph": {
            "nodes": [
                {
                    "id": "dock_station",
                    "class": "dock",
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "size": {"x": 0.5, "y": 0.5, "z": 0.2},
                    "frame": "map",
                }
            ],
            "edges": [],
        },
        "robots": {
            "desktop_pet_001": {
                "connection_state": {
                    "status": "connected",
                    "transport": "local",
                    "host": "localhost",
                    "port": 0,
                    "last_heartbeat": "2026-03-20T00:00:00Z",
                    "last_error": None,
                    "reconnect_attempts": 0,
                },
                "robot_pose": {
                    "frame": "desk",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": 0.0,
                    "stamp": "2026-03-20T00:00:00Z",
                },
                "nav_state": {"mode": "idle", "status": "idle"},
            }
        },
        "objects": {
            "apple": {"type": "fruit", "position": {"x": 0, "y": 0, "z": 0}}
        },
    }


def test_watchdog_health_refresh_persists_connection_state(tmp_path: Path) -> None:
    env_file = tmp_path / "ENVIRONMENT.md"
    save_environment_doc(env_file, _base_environment())
    action_file = tmp_path / "ACTION.md"
    action_file.write_text("", encoding="utf-8")

    with load_driver("go2_edu", gui=False) as driver:
        driver.load_scene({"apple": {"type": "fruit", "position": {"x": 0, "y": 0, "z": 0}}})
        _poll_once(driver, action_file, env_file)

    updated = load_environment_doc(env_file)
    go2_state = updated["robots"]["go2_edu_001"]["connection_state"]
    assert go2_state["status"] == "connected"
    assert go2_state["transport"] == "ssh"
    assert go2_state["host"] == "192.168.1.23"
    assert go2_state["last_heartbeat"] is not None
    assert updated["robots"]["desktop_pet_001"]["connection_state"]["status"] == "connected"
    assert updated["scene_graph"]["nodes"][0]["id"] == "dock_station"
    assert updated["objects"]["apple"]["type"] == "fruit"


def test_explicit_disconnect_and_reconnect_actions_update_runtime_connection_state(tmp_path: Path) -> None:
    env_file = tmp_path / "ENVIRONMENT.md"
    save_environment_doc(env_file, _base_environment())

    with load_driver("go2_edu", gui=False) as driver:
        driver.load_scene({"apple": {"type": "fruit", "position": {"x": 0, "y": 0, "z": 0}}})

        disconnect_action = _write_action(
            tmp_path,
            {
                "action_type": "disconnect_robot",
                "parameters": {"robot_id": "go2_edu_001"},
                "status": "pending",
            },
        )
        _poll_once(driver, disconnect_action, env_file)

        disconnected = load_environment_doc(env_file)
        disconnect_state = disconnected["robots"]["go2_edu_001"]["connection_state"]
        assert disconnect_state["status"] == "disconnected"
        assert disconnect_action.read_text(encoding="utf-8").strip() == ""

        reconnect_action = _write_action(
            tmp_path,
            {
                "action_type": "reconnect_robot",
                "parameters": {"robot_id": "go2_edu_001"},
                "status": "pending",
            },
        )
        _poll_once(driver, reconnect_action, env_file)

    updated = load_environment_doc(env_file)
    reconnect_state = updated["robots"]["go2_edu_001"]["connection_state"]
    assert reconnect_state["status"] == "connected"
    assert reconnect_state["last_error"] is None
    assert reconnect_state["last_heartbeat"] is not None
    assert reconnect_action.read_text(encoding="utf-8").strip() == ""
