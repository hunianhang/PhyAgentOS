# Environment State

Auto-updated by HAL Watchdog and/or side-loaded perception services.
This file stores a multi-agent environment snapshot in a structured format.

Notes:
- `robots.<robot_id>.robot_pose` stores each robot's current pose state.
- `robots.<robot_id>.nav_state` stores each robot's navigation/task runtime state.
- `objects` is the object-level world state used by current HAL drivers.

```json
{
  "schema_version": "oea.environment.v1",
  "updated_at": "2026-03-17T10:20:30Z",
  "scene_graph": {
    "nodes": [
      {
        "id": "obj_red_apple",
        "class": "apple",
        "object_key": "red_apple",
        "center": {"x": 0.05, "y": 0.05, "z": 0.75},
        "size": {"x": 0.06, "y": 0.06, "z": 0.06},
        "confidence": 0.96
      },
      {
        "id": "obj_blue_cup",
        "class": "cup",
        "object_key": "blue_cup",
        "center": {"x": -0.10, "y": 0.03, "z": 0.78},
        "size": {"x": 0.08, "y": 0.08, "z": 0.12},
        "confidence": 0.94
      }
    ],
    "edges": [
      {"source": "obj_red_apple", "relation": "ON", "target": "furniture_table"},
      {"source": "obj_blue_cup", "relation": "ON", "target": "furniture_table"},
      {"source": "obj_red_apple", "relation": "CLOSE_TO", "target": "obj_blue_cup"}
    ]
  },
  "robots": {
    "go2_edu_001": {
      "robot_pose": {
        "frame": "map",
        "x": 1.23,
        "y": -0.45,
        "z": 0.0,
        "yaw": 1.57,
        "stamp": "2026-03-17T10:20:30Z"
      },
      "nav_state": {
        "mode": "navigating",
        "status": "running",
        "goal_id": "nav_goal_001",
        "goal": {"x": 2.0, "y": 1.0, "yaw": 0.0},
        "path_progress": 0.62,
        "last_error": null
      }
    },
    "desktop_pet_001": {
      "robot_pose": {
        "frame": "desk",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "stamp": "2026-03-17T10:20:29Z"
      },
      "nav_state": {
        "mode": "idle",
        "status": "idle"
      }
    }
  },
  "objects": {
    "red_apple": {
      "type": "fruit",
      "color": "red",
      "location": "table",
      "position": {"x": 5, "y": 5, "z": 0}
    },
    "blue_cup": {
      "type": "container",
      "color": "blue",
      "location": "table",
      "position": {"x": -10, "y": 3, "z": 0},
      "state": "empty"
    }
  }
}
```
