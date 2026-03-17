## 整体框架

### 1. NanoBot (大脑/规划层)

- **职责**：大模型（LLM/VLM）驻留地。负责解析用户的自然语言指令，进行任务分解（Task Planning）。
- **动作流**：不关心具体的 $X, Y$ 坐标，只输出语义级指令，例如调用 `ActionLayer.navigate_to("冰箱")` 或 `ActionLayer.explore_room()`。

### 2. Action 层 (技能/中间件层)

- **职责**：承上启下。将大脑（NanoBot）传来的“高阶语义目标”翻译为具体的“空间几何目标”，并调用底层的导航算法规划出连续的动作指令。
- **内部模块集成**：
  - **场景图缓存池 (Scene Graph Context)**：维持与后台感知系统的通信，随时可以查询 `Env Graph`。
  - **语义导航动作 (Semantic Nav Action)**：当收到“去冰箱”指令时，在此层查询场景图拿到冰箱的 BBox 坐标，结合避障算法（Nav2 的 Planner Server）生成全局和局部路径。
  - **指令下发**：将局部路径规划器计算出的通用控制指令（标准化的线速度 $v$、角速度 $\omega$）发送给下方的 Embodied 层。

### 3. Embodied 层 (硬件适配与接口层)

- **职责**：设备的“驱动器”与“翻译官”。
- **上行数据 (感知提取)**：通过各个机器人的专有 SDK 拉取 RGB-D 和雷达数据，统一封装为标准格式（ROS 2 的 `Image` 和 `PointCloud2` / `LaserScan` 消息，需额外统一定义），供侧边的建图模块消费。
- **下行控制 (动作执行)**：接收 Action 层传来的标准化速度/位姿指令，**映射并调用具体机器人的 SDK 控制接口**（例如：转化为宇树机器狗的 `HighCmd` 或移动底盘的串行控制帧），最终驱动电机。

### 4. 侧载感知与记忆引擎 (Side-loaded Perception Engine)

- **职责**：作为一个独立于控制链路之外的守护进程（Daemon）运行。
- **工作流**：
  1. 从 Embodied 层获取标准化的传感器流。
  2. 运行 2D 分割、3D BBox 投影融合、SLAM 里程计和图匹配。
  3. 实时更新局部的 Scene Graph 并序列化。
  4. 为 Action 层提供实时的 TF（坐标系转换）支持和地图查询 API。

## 具体细节

### 一、 Embodied层：硬件抽象与数据双向适配

这一层是系统的“基建”，负责将不同厂商的非标准机器翻译为 ROS 2 标准协议。

#### 1. 传感器数据上行 (SDK -> ROS 2)

需要开发几个独立的适配器节点（Adapter Nodes），通过各家机器人的 SDK 拉取数据，并**严格打上统一的时钟戳（`header.stamp`）**发布到 ROS 2 网络中：

- **RGB-D 适配器**：
  - 发布 `/camera/color/image_raw` (`sensor_msgs/Image`)
  - 发布 `/camera/depth/image_aligned` (`sensor_msgs/Image`，必须是与 RGB 对齐的深度图)
  - 发布 `/camera/color/camera_info` (`sensor_msgs/CameraInfo`，包含内参 $K$ 矩阵，极度关键)
- **雷达适配器 (针对 Mid-360)**：
  - 通过 Livox SDK 发布自定义点云，并立即通过 `livox_ros2_driver` 转化为 `/livox/lidar` (`sensor_msgs/PointCloud2`)。
- **底盘里程计适配器**：
  - 读取轮式里程计或 IMU 数据，发布底盘的 `/odom` 以及 `odom` -> `base_link` 的 TF 变换。

#### 2. 控制指令下行 (ROS 2 -> SDK)

- 编写一个 `cmd_vel_listener` 节点，订阅 `/cmd_vel` (`geometry_msgs/Twist`)。
- 将标准的线速度 $v_x, v_y$ 和角速度 $\omega_z$ 提取出来，调用机器人 SDK 的运动控制接口（例如大疆 EP 的 `chassis.drive_speed()` 或宇树的 `high_cmd`）。

------

### 二、 侧载感知引擎：3D语义场景图的流式构建

这是本方案的核心算力区，作为一个守护进程静默运行，包含四个核心流水线节点：

#### 1. 几何定位与建图 (Geometry SLAM Pipeline)

为了兼顾 Mid-360 的 3D 特性和 `slam_toolbox` 的 2D 需求：

- **3D 里程计**：运行 `FAST-LIO2`，接收 Mid-360 点云和 IMU，输出高精度的 `map` -> `odom` TF 树，以及去畸变的 3D 点云。
- **2D 降维**：使用 `pointcloud_to_laserscan` 节点，将 FAST-LIO2 输出的 3D 点云按机器人高度（如 0.2m - 1.5m）切片，投影为 `/scan` (`sensor_msgs/LaserScan`)。
- **全局栅格图**：将 `/scan` 喂给 `slam_toolbox`，生成用于避障和导航的 2D 占据栅格地图 (`/map`)。

#### 2. 2D 语义实例分割 (Semantic Segmentation)

- 订阅 `/camera/color/image_raw`。
- 运行轻量级的高频分割模型（推荐 YOLOv8-Seg TensorRT 加速版）。
- 输出自定义消息 `SemanticMaskArray`，包含每个识别到的物体类别（如 `sofa`, `cup`）、置信度以及 2D 轮廓像素 Mask。

#### 3. 2D 到 3D 的视锥升维与融合 (3D BBox Fuser)

**此节点是工程稳定性的分水岭。**

- **深度反投影**：提取 2D Mask 内的所有像素坐标 $(u, v)$，查找对应的深度值 $d$。利用相机内参 $(c_x, c_y, f_x, f_y)$ 计算相机坐标系下的 3D 点云：

  $$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$

- **时空对齐**：查询 TF 树，获取当前时间戳下 `camera_link` 到 `map`（全局地图）的变换矩阵，将这些 3D 点转换到全局坐标系中。

- **点云降噪与 BBox 拟合**：使用 PCL（Point Cloud Library）对全局点云簇进行体素滤波（Voxel Grid）和统计滤波（Statistical Outlier Removal）去除噪点。随后计算点云的 AABB（轴对齐包围盒）或 OBB，获取中心点 $(X, Y, Z)$ 和长宽高。

- **多目标跟踪 (MOT)**：对比上一帧的 BBox，利用 3D IoU 和卡尔曼滤波分配唯一 ID，防止同一物体在连续帧中闪烁。

#### 4. 场景图构建器 (Scene Graph Builder)

将带 ID 的 3D BBox 转化为 VirtualHome 格式：

- **节点 (Nodes)**：直接映射 BBox 的类别、中心坐标和尺寸。
- **边 (Edges)**：设定启发式几何规则。
  - 如果 BBox A 的 Z 轴底面接触 BBox B 的 Z 轴顶面，且 XY 投影重合 -> `(A, ON, B)`。
  - 如果 BBox A 与 BBox B 欧氏距离小于 0.5m -> `(A, CLOSE_TO, B)`。
- **持久化**：将内存中的 Graph 实时序列化为 JSON。

------

### 三、 Action层：技能封装与 Nav2 对接

这一层向 OpenClaw 的 LLM 暴露高级 API，并向下调用 Nav2 实现闭环。

#### 1. Nav2 导航栈配置

在 Action 层内部，启动 ROS 2 Nav2 栈：

- **Global Costmap（全局代价地图）**：基于 `slam_toolbox` 的静态地图构建，用于计算从 A 房间到 B 房间的全局路径。
- **Local Costmap（局部代价地图）**：直接订阅 Mid-360 降维后的 `/scan`，用于实时避开突然出现的人或动态障碍物。

#### 2. Python 技能 API 实现

在 Action 层中编写 Python 类 `SemanticNavigator`，桥接 OpenClaw：

Python

```
# Action层暴露给 OpenClaw 的接口示例
class SemanticNavigator:
    def __init__(self):
        # 初始化与底层 Scene Graph 节点的 ROS 2 通信客户端
        self.graph_client = GraphQueryClient()
        self.nav2_client = BasicNavigator() # ROS 2 Nav2 的 Python API

    def navigate_to_target(self, target_class: str) -> str:
        """供 OpenClaw 调用的 Action"""
        # 1. 查询 Scene Graph
        nodes = self.graph_client.query_nodes_by_class(target_class)
        if not nodes:
            return f"Action Failed: 场景中未发现 {target_class}。"
        
        target_node = nodes[0] # 假设取最近的一个
        center_x, center_y = target_node.bbox.center[0], target_node.bbox.center[1]
        
        # 2. 计算安全交互点 (向外推演 0.5 米，防止机器人撞上物体)
        # 此处需要一段几何逻辑，根据目标 BBox 的边缘和机器人当前位置，计算一个可达的 Pose
        goal_pose = self._calculate_approach_pose(center_x, center_y, target_node.bbox.size)

        # 3. 发送给 Nav2
        self.nav2_client.goToPose(goal_pose)
        
        # 4. 阻塞等待或异步回调
        while not self.nav2_client.isTaskComplete():
            pass # 可以加入超时逻辑和避障重试逻辑
            
        result = self.nav2_client.getResult()
        if result == TaskResult.SUCCEEDED:
            return f"Action Success: 已到达 {target_class} 附近。"
        else:
            return "Action Failed: 导航受阻。"
```

------

### 四、 地图重定位与更新机制 (Relocalization)

当机器人关机重启后，如何在现有的 Scene Graph 地图中找到自己的位置？

1. **冷启动与局部感知**：机器人开机，Embodied 层开始推流。感知引擎运行，根据当前相机的视野，生成一个**“局部临时场景图 (Local Graph)”**。
2. **提取语义锚点 (Semantic Anchors)**：从保存的全局 Scene Graph JSON 中，提取 `Furniture`（如床、柜子、沙发）作为静态锚点。
3. **图匹配对齐 (Graph Matching)**：
   - Action 层提供一个 `localize()` 技能。
   - 该技能指示机器人原地旋转 360 度。
   - 使用 3D 匈牙利匹配算法（基于类别和相对拓扑关系），将“局部临时场景图”中的节点与“全局场景图”中的节点进行匹配。
4. **TF 树校正**：匹配成功后，计算出当前位姿与地图原点的变换矩阵，通过 Nav2 的 `/initialpose` 话题发布出去，瞬间完成 `map` 和 `odom` 的对齐。
5. **增量更新**：对齐后，如果发现原本在桌子上的杯子不见了，图构建器会降低该节点的置信度，最终将其从当前内存的 Graph 中剔除。