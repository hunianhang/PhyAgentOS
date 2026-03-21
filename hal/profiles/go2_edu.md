# Robot Embodiment Declaration

This file describes the physical capabilities and constraints of the connected robot.
The Critic Agent reads this file to validate whether proposed actions are safe and feasible.

## Identity

- **Name**: Unitree Go2 EDU
- **Type**: Quadruped mobile robot

## Sensors

- **RGB-D**: Optional front camera pipeline via adapter nodes
- **LiDAR**: Mid-360 / 4D LiDAR compatible
- **Odometry**: IMU + locomotion odometry

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `semantic_navigate` | `robot_id, target_ref, goal_pose, approach_distance, timeout_s` | Navigate to a semantic target using scene graph lookup and Nav2-compatible goals |
| `localize` | `robot_id, mode, timeout_s` | Trigger relocalization workflow |
| `stop` | `robot_id` | Stop the current navigation task |
| `connect_robot` | `robot_id` | Establish the control connection to the robot |
| `disconnect_robot` | `robot_id` | Close the current control connection |
| `reconnect_robot` | `robot_id` | Reset and restore the control connection |
| `check_connection` | `robot_id` | Run the heartbeat and update runtime connection state |

## Connection

- **Transport**: ssh
- **Host**: 192.168.1.23
- **Port**: 22
- **User**: robot
- **Auth**: key
- **Remote Control API**: /usr/local/bin/robot_ctl
- **Heartbeat Command**: echo ok
- **Reconnect Policy**: auto

## Navigation Capabilities

- **Frames**: `map`, `odom`, `base_link`, `camera_link`, `lidar`
- **Max linear speed**: 1.5 m/s
- **Max angular speed**: 1.0 rad/s
- **Minimum obstacle clearance**: 0.5 m
- **Relocalization support**: yes
- **ROS2 command channels**: `/cmd_vel`, `/navigate_to_pose`, `/initialpose`

## Runtime Protocol

- **Connection channel**: `robots.go2_edu_001.connection_state`
- **Pose channel**: `robots.go2_edu_001.robot_pose`
- **Navigation channel**: `robots.go2_edu_001.nav_state`
- **Health owner**: `hal_watchdog.py` triggers connect on startup and periodic `health_check()`.
- **Reconnect behavior**: driver may auto-reconnect according to the declared policy.

## Physical Constraints

- **Operating area**: bounded by environment map or geofence declared in `ENVIRONMENT.md`
- **Collision policy**: stop and mark `nav_state.last_error` on unrecoverable obstruction
