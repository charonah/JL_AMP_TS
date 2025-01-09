import pybullet as p
import pybullet_data
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("/home/ubuntu/Documents/code/rl_ts_amp_jl/src/AMP_for_hardware/resources/robots/jl1/urdf/ysc2gotest.urdf")
num_joints = p.getNumJoints(robot)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    print(f"Joint {i}: Name={joint_info[1].decode('utf-8')}")
# 获取根部位置
root_pos, root_rot = p.getBasePositionAndOrientation(robot)
print(f"Initial Root Position: {root_pos}")
print(f"Initial Root Rotation: {root_rot}")

# 假设需要将机器人提升0.04米
SIM_ROOT_OFFSET = np.array([0, 0, 0.27])
print(f"SIM_ROOT_OFFSET set to: {SIM_ROOT_OFFSET}")


# 假设SIM_TOE_JOINT_IDS已经确定
SIM_TOE_JOINT_IDS = [4, 8, 12, 16]  # 替换为新的脚趾关节ID

SIM_TOE_OFFSET_LOCAL = []
for joint_id in SIM_TOE_JOINT_IDS:
    link_state = p.getLinkState(robot, joint_id, computeForwardKinematics=True)
    print(link_state)
    foot_pos = np.array(link_state[4])  # 世界坐标系中的位置
    foot_ori = link_state[5]  # 世界坐标系中的旋转
    # 假设脚趾在局部坐标系中的偏移可以通过逆变换获得
    # 这里简化为脚趾在关节坐标系中的偏移为foot_pos - root_pos
    # 实际情况可能需要根据机器人结构调整
    local_offset = foot_pos - np.array(p.getBasePositionAndOrientation(robot)[0])
    SIM_TOE_OFFSET_LOCAL.append(local_offset)

print(f"SIM_TOE_OFFSET_LOCAL: {SIM_TOE_OFFSET_LOCAL}")
p.disconnect()
