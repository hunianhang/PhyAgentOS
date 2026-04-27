# Robot Embodiment Declaration

This profile describes a LeRobot-based single robot arm controlled via the HuggingFace LeRobot SDK over serial USB, with optional Solo-CLI integration for ML pipeline operations.

Supported hardware: **SO-101**, **Koch v1.1**, **SO-100**.

## Identity

- **Name**: SO-101 / Koch v1.1 (LeRobot + Solo-CLI)
- **Type**: 6-DOF tabletop manipulator arm with gripper
- **Runtime Topology**: OEA/HAL on host PC → LeRobot SDK → serial USB → Feetech STS3215 servo bus
- **Hardware Source**: [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), [LeRobot](https://huggingface.co/docs/lerobot/en/so101), [Solo-CLI](https://github.com/GetSoloTech/solo-cli)

## Joints

| Index | Joint Name | Description |
|-------|------------|-------------|
| 1 | `shoulder_pan` | Base rotation |
| 2 | `shoulder_lift` | Shoulder up/down |
| 3 | `elbow_flex` | Elbow bend |
| 4 | `wrist_flex` | Wrist pitch |
| 5 | `wrist_roll` | Wrist rotation (continuous) |
| 6 | `gripper` | Gripper open/close |

## Sensors

- **Joint feedback**: Position feedback from Feetech STS3215 servos via bus protocol
- **Camera**: Optional USB camera(s) configurable via `cameras` driver config key (OpenCV backend)

## Supported Actions

### Direct Joint Control

| Action | Parameters | Description |
|--------|-----------|-------------|
| `connect_robot` | `robot_id` (optional) | Connect to the arm via serial USB and initialize the LeRobot SDK |
| `check_connection` | `robot_id` (optional) | Run health check by reading servo observation |
| `disconnect_robot` | `robot_id` (optional) | Disconnect from the arm |
| `set_joint_targets` | `joints` object, `robot_id` (optional) | Send position targets for one or more joints (e.g. `{"shoulder_pan": 45.0, "elbow_flex": -30.0}`) |
| `set_gripper` | `value`, `robot_id` (optional) | Set gripper target value (0 = closed, positive = open) |
| `get_joint_positions` | `robot_id` (optional) | Read current joint positions from servo feedback |
| `go_to_rest` | `robot_id` (optional) | Move all joints to zero / rest position |

### Solo-CLI Pipeline Commands

These shell out to the `solo robo` CLI for ML pipeline operations. Requires `solo-cli` installed.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `solo_calibrate` | `auto_confirm` (bool), `timeout_s` (int) | Run `solo robo --calibrate all` |
| `solo_teleop` | `auto_confirm`, `timeout_s` | Run `solo robo --teleop` for leader-follower teleoperation |
| `solo_record` | `auto_confirm`, `timeout_s` | Run `solo robo --record` to record a dataset |
| `solo_train` | `auto_confirm`, `timeout_s` | Run `solo robo --train` to train ACT/SmolVLA policy |
| `solo_inference` | `auto_confirm`, `timeout_s` | Run `solo robo --inference` for VLA inference |
| `solo_replay` | `auto_confirm`, `timeout_s` | Run `solo robo --replay` to replay a recorded episode |

### LeRobot Native CLI Commands

Direct calls to the LeRobot Python CLI, useful when solo-cli is not installed.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `lerobot_calibrate` | `target` ("follower"/"leader"), `timeout_s` | Run `lerobot-calibrate` for the specified arm |
| `lerobot_teleop` | `timeout_s` | Run `lerobot-teleoperate` (requires leader arm configured) |

## Physical Constraints

- **Arm type**: 6-DOF serial manipulator, tabletop-mounted, no mobile base
- **Servos**: Feetech STS3215 (follower: 1/345 gearing; leader: mixed gearing per joint)
- **Workspace**: ~30 cm reach radius
- **Gripper**: Single parallel-jaw gripper (joint 6)
- **Safety**: `max_step_deg` limits per-step joint movement; calibration required before first use

## Connection

- **Transport**: Serial USB (Feetech bus servo adapter, typically `/dev/ttyACM0`)
- **Preferred driver config keys**: `arm_type`, `port`, `robot_id`, `leader_port`, `leader_type`, `leader_id`, `loop_hz`, `use_degrees`, `max_step_deg`, `cameras`, `reconnect_policy`
- **Watchdog CLI**: `python hal/hal_watchdog.py --driver so101_follower --driver-config <json>`
- **Environment fallback**: `LEROBOT_ARM_TYPE`, `LEROBOT_ARM_PORT`, `LEROBOT_ROBOT_ID`, `LEROBOT_LEADER_PORT`, `LEROBOT_LEADER_TYPE`, `LEROBOT_LOOP_HZ`, `LEROBOT_MAX_STEP_DEG`

## Setup Prerequisites

Before using this driver, the arm must be assembled and configured:

1. **Install LeRobot SDK**: `pip install lerobot` (or `pip install -e ".[feetech]"` from source)
2. **Install Solo-CLI** (optional): `pip install solo-cli` or clone from [GitHub](https://github.com/GetSoloTech/solo-cli)
3. **Find USB port**: Run `lerobot-find-port` to discover the serial device
4. **Set motor IDs**: Run `lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>` (or `solo robo --motors all`)
5. **Calibrate**: Run `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<name>` (or `solo robo --calibrate all`)

See [SO-101 Assembly Guide](https://huggingface.co/docs/lerobot/en/so101) for full details.

## Runtime Protocol

- **Connection channel**: `robots.<robot_id>.connection_state`
- **Pose channel**: `robots.<robot_id>.robot_pose` (joint positions dict with heartbeat timestamp)
- **Navigation channel**: `robots.<robot_id>.nav_state` (action status)
- **Health owner**: `hal_watchdog.py` via periodic `health_check()`

## Notes

- This driver wraps the upstream LeRobot Python API (`SO101Follower`, `KochFollower`, `SO100Follower`) and does not re-implement servo communication.
- Teleoperation requires a **leader arm** (separate physical arm). Configure `leader_port`, `leader_type`, and `leader_id` in the driver config.
- The Solo-CLI (`solo robo`) provides an integrated workflow: motors → calibrate → teleop → record → train → inference. The driver can trigger these steps as actions.
- Higher-level behaviors (pick-and-place, visual servoing) should be composed by OEA workflow layers, not embedded in this HAL driver.
