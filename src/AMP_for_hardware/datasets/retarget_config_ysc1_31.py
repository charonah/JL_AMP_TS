import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jl1/urdf/ysc2gotest.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/mocap_motions_ysc1_31".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

# 为什么是为个值
REF_POS_SCALE = 0.9
# 期望root高度
INIT_POS = np.array([0, 0, 0.3])
INIT_ROT = np.array([0, 0, 0, 1.0])
# lf,rf,lr,rr的顺序　
SIM_TOE_JOINT_IDS = [8, 4, 16, 12] #pybullet中对应joint的index，可以在set_pose中打印出来
SIM_HIP_JOINT_IDS = [5, 1, 13, 9]
# retarget 后的root高度要减去这个offset,这个keypoint的数据集高度在0.367左右
# 所以root高度要减去0.367-0.04=0.327，和上面的init_pose差不多
SIM_ROOT_OFFSET = np.array([0, 0, -0.07])
# feet相对于hip的局部偏移量offset，　fl,fr,rl,rr,
# 这个值是不是并不需要很准，a1的hip offset_y = 0.047,thight + 0.0838
SIM_TOE_OFFSET_LOCAL = [
    np.array([-0.03, 0.12, -0.0]),
    np.array([-0.03, -0.12, -0.0]),
    np.array([0.03, 0.012, -0.0]),
    np.array([0.03, -0.12, -0.0])
]
# radius = 0.02
TOE_HEIGHT_OFFSET = 0.02169
#求IK时会使用，站立时对应高度和位置　
DEFAULT_JOINT_POSE = np.array([0.1, 0.66, -1.5, -0.1, 0.66, -1.5, 0.1, 0.66, -1.5, -0.1, 0.66, -1.5])
# 12个关节的kd,用于计算IK时使用
JOINT_DAMPING = [4, 1, 1,
                 4, 1, 1,
                 4, 1, 1,
                 4, 1, 1]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

# link name ,urdf中的link name
FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "pace0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 162, 201, 1
    ],
    [
        "pace1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 201, 400, 1
    ],
    [
        "pace2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 400, 600, 1
    ],
    [
        "trot0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 448, 481, 1
    ],
    [
        "trot1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 400, 600, 1
    ],
    [
        "trot2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 480, 663, 1
    ],
    [
        "canter0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 430, 480, 1
    ],
    [
        "canter1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 380, 430, 1
    ],
    [
        "canter2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 480, 566, 1
    ],
    [
        "right_turn0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 1085, 1124, 1.5
    ],
    [
        "right_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 560, 670, 1.5
    ],
    [
        "left_turn0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 2404, 2450, 1.5
    ],
    [
        "left_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 120, 220, 1.5
    ]
]