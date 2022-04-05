import numpy as np
from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config as habitat_get_config
from habitat.config.default import (
    CONFIG_FILE_SEPARATOR,
    DEFAULT_CONFIG_DIR,
)

__all__ = [
    'get_config',
    'finetune_config'
]


_C = habitat_get_config()
_C.defrost()

# -----------------------------------------------------------------------------
# GT POSE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GT_POSE_SENSOR = CN()
_C.TASK.GT_POSE_SENSOR.TYPE = "GTPoseSensor"
# -----------------------------------------------------------------------------
# FAKE SPL
# -----------------------------------------------------------------------------
_C.TASK.FAKE_SPL = CN()
_C.TASK.FAKE_SPL.TYPE = 'FakeSPL'
# -----------------------------------------------------------------------------
# AGENT POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.AGENT_POSITION_SENSOR = CN()
_C.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position_sensor"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True


# -----------------------------------------------------------------------------
# ACTIVE NEURAL SLAM (ANS)
# -----------------------------------------------------------------------------
_C.ANS = CN()


# -----------------------------------------------------------------------------
# MAP BUILDER (Global)
# -----------------------------------------------------------------------------
_C.ANS.MAP_BUILDER = CN()
_C.ANS.MAP_BUILDER.MAP_RES = 0.015 # meters per map cell
_C.ANS.MAP_BUILDER.MAP_WIDTH = 640  # map width cells
_C.ANS.MAP_BUILDER.MAP_HEIGHT = 640 # map height cells
_C.ANS.MAP_BUILDER.TRUNC_DEPTH_MIN = 0.12
_C.ANS.MAP_BUILDER.TRUNC_DEPTH_MAX = 5.05 # the depth far than this number is considered as invalid range (unexplored)
_C.ANS.MAP_BUILDER.CLIP_BORDER = 20 # Clip depth map border (pixel)
_C.ANS.MAP_BUILDER.SMOOTH_ITER = 2
_C.ANS.MAP_BUILDER.SMOOTH_WINDOW = 3
_C.ANS.MAP_BUILDER.FREE_THRES = 0.2 # height threshold, below this nubmer is considered as free spaces
_C.ANS.MAP_BUILDER.OBSTACLE_THRES = 1.5 # free_thres ~ obstacle_thres -> obstacle
_C.ANS.MAP_BUILDER.FREE_SMOOTH_ITER = 3
_C.ANS.MAP_BUILDER.FREE_SMOOTH_WINDOW = 3
_C.ANS.MAP_BUILDER.OBSTACLE_SMOOTH_ITER = 2
_C.ANS.MAP_BUILDER.OBSTACLE_SMOOTH_WINDOW = 3

# -----------------------------------------------------------------------------
# MAP BUILDER (Local)
# -----------------------------------------------------------------------------
_C.ANS.MAP_BUILDER.LOCAL = CN()
_C.ANS.MAP_BUILDER.LOCAL.MAP_RES = 0.06 # meters per map cell
_C.ANS.MAP_BUILDER.LOCAL.MAP_WIDTH = 180
_C.ANS.MAP_BUILDER.LOCAL.MAP_HEIGHT = 180
_C.ANS.MAP_BUILDER.LOCAL.TRUNC_DEPTH_MIN = 0.12
_C.ANS.MAP_BUILDER.LOCAL.TRUNC_DEPTH_MAX = 5.05 # the depth far than this number is considered as invalid range (unexplored)
_C.ANS.MAP_BUILDER.LOCAL.CLIP_BORDER = 20 # Clip depth map border (pixel)
_C.ANS.MAP_BUILDER.LOCAL.SMOOTH_ITER = 1
_C.ANS.MAP_BUILDER.LOCAL.SMOOTH_WINDOW = 3
_C.ANS.MAP_BUILDER.LOCAL.FREE_THRES = 0.2 # height threshold, below this nubmer is considered as free spaces
_C.ANS.MAP_BUILDER.LOCAL.OBSTACLE_THRES = 1.5 # free_thres ~ obstacle_thres -> obstacle
_C.ANS.MAP_BUILDER.LOCAL.FREE_SMOOTH_ITER = 3
_C.ANS.MAP_BUILDER.LOCAL.FREE_SMOOTH_WINDOW = 3
_C.ANS.MAP_BUILDER.LOCAL.OBSTACLE_SMOOTH_ITER = 1
_C.ANS.MAP_BUILDER.LOCAL.OBSTACLE_SMOOTH_WINDOW = 3
# -----------------------------------------------------------------------------
# Pose Estimator
# -----------------------------------------------------------------------------
_C.ANS.PE = CN()
_C.ANS.PE.INPUT_SHAPE = (128, 128, 4)
_C.ANS.PE.MAP_SHAPE = (64, 64, 1)
_C.ANS.PE.MIN_POSE = [-0.1, -0.1, -0.1]
_C.ANS.PE.MAX_POSE = [0.1, 0.1, 0.1]
_C.ANS.PE.NUM_SAMPLES = 256
_C.ANS.PE.SEQ_LEN = 4
_C.ANS.PE.EMBED_SIZE = 64
_C.ANS.PE.ENCODE_SIZE = 512
_C.ANS.PE.RECURRENT_UNIT = 256
_C.ANS.PE.DECODE_SIZE = 3
_C.ANS.PE.POSE_REPEAT = 42
_C.ANS.PE.RANDOM_PERMUTE = True
_C.ANS.PE.ENABLE_ATTEN = True
# --- pretrain ---
_C.ANS.PE.PRETRAIN = CN()
_C.ANS.PE.PRETRAIN.LEARNING_RATE = 3e-4
_C.ANS.PE.PRETRAIN.BUFFER_SIZE = 60000
_C.ANS.PE.PRETRAIN.MIN_BUFFER = 3000
_C.ANS.PE.PRETRAIN.N_STEPS = 1000
_C.ANS.PE.PRETRAIN.N_GRADSTEPS = 100
_C.ANS.PE.PRETRAIN.BATCH_SIZE = 64
_C.ANS.PE.PRETRAIN.POSE_COEF = 0.2
_C.ANS.PE.PRETRAIN.DELTA_COEF = 1.0
_C.ANS.PE.PRETRAIN.REG_COEF = 4e-6
_C.ANS.PE.PRETRAIN.YAW_COEF = 0.36
_C.ANS.PE.PRETRAIN.ENT_COEF = 1e-3
_C.ANS.PE.PRETRAIN.TRAIN_SEQ_LEN = 50
_C.ANS.PE.PRETRAIN.MAX_GRAD_NORM = 0.5
# --- train ---
_C.ANS.PE.TRAIN = CN()
_C.ANS.PE.TRAIN.LEARNING_RATE = 3e-5
_C.ANS.PE.TRAIN.BUFFER_SIZE = 30000
_C.ANS.PE.TRAIN.MIN_BUFFER = 3000
_C.ANS.PE.TRAIN.N_STEPS = 1000
_C.ANS.PE.TRAIN.N_GRADSTEPS = 120
_C.ANS.PE.TRAIN.BATCH_SIZE = 8
_C.ANS.PE.TRAIN.POSE_COEF = 0.2
_C.ANS.PE.TRAIN.DELTA_COEF = 1.0
_C.ANS.PE.TRAIN.REG_COEF = 4e-6
_C.ANS.PE.TRAIN.YAW_COEF = 0.36
_C.ANS.PE.TRAIN.ENT_COEF = 1e-3
_C.ANS.PE.TRAIN.TRAIN_SEQ_LEN = 60
_C.ANS.PE.TRAIN.MAX_GRAD_NORM = 0.5

# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------
_C.ANS.PLANNER = CN()
_C.ANS.PLANNER.UNEXPLORED_WEIGHTS = 2 # (astar weights)
_C.ANS.PLANNER.FREE_WEIGHTS = 1 # (astar weights)
_C.ANS.PLANNER.OBSTACLE_WEIGHTS = 1e+5 # (astar weights)
_C.ANS.PLANNER.OBSTACLE_RANGE = 10 # (astar weights)
_C.ANS.PLANNER.TOLERANCE = 1 # (astar weights)
_C.ANS.PLANNER.SUB_GOAL_DISTANCE = 1.0 # (meter)
_C.ANS.PLANNER.CROP_PLAN_RANGE = True
_C.ANS.PLANNER.PAD_PLAN_RANGE = 3 # (meter)
# -----------------------------------------------------------------------------
# Local Policy
# -----------------------------------------------------------------------------
_C.ANS.POLICY = CN()
_C.ANS.POLICY.EMBED_SIZE = 256
_C.ANS.POLICY.RECURRENT_UNIT = 256
_C.ANS.POLICY.FEATURES = CN()
_C.ANS.POLICY.FEATURES.DISTANCE = CN()
_C.ANS.POLICY.FEATURES.DISTANCE.MIN = 0.1
_C.ANS.POLICY.FEATURES.DISTANCE.MAX = 2
_C.ANS.POLICY.FEATURES.DISTANCE.COUNT = 10
_C.ANS.POLICY.FEATURES.DISTANCE.EMBED_SIZE = 32
_C.ANS.POLICY.FEATURES.DISTANCE.USE_LOG_SCALE = True
_C.ANS.POLICY.FEATURES.ANGLE = CN()
_C.ANS.POLICY.FEATURES.ANGLE.MIN = -np.pi
_C.ANS.POLICY.FEATURES.ANGLE.MAX = np.pi
_C.ANS.POLICY.FEATURES.ANGLE.COUNT = 72
_C.ANS.POLICY.FEATURES.ANGLE.EMBED_SIZE = 32
_C.ANS.POLICY.FEATURES.ANGLE.USE_LOG_SCALE = False
_C.ANS.POLICY.FEATURES.TIME = CN()
_C.ANS.POLICY.FEATURES.TIME.MIN = 0
_C.ANS.POLICY.FEATURES.TIME.MAX = 10
_C.ANS.POLICY.FEATURES.TIME.COUNT = 11
_C.ANS.POLICY.FEATURES.TIME.EMBED_SIZE = 32
_C.ANS.POLICY.FEATURES.TIME.USE_LOG_SCALE = False
# -----------------------------------------------------------------------------
# PPO
# -----------------------------------------------------------------------------
_C.ANS.PPO = CN()
_C.ANS.PPO.LEARNING_RATE = 3e-4
_C.ANS.PPO.N_STEPS = 1024
_C.ANS.PPO.BATCH_SIZE = 512
_C.ANS.PPO.N_SUBEPOCHS = 8
_C.ANS.PPO.GAMMA = 0.99
_C.ANS.PPO.GAE_LAMBDA = 0.95
_C.ANS.PPO.CLIP_RANGE = 0.2
_C.ANS.PPO.CLIP_RANGE_VF = None
_C.ANS.PPO.ENT_COEF = 1e-2
_C.ANS.PPO.VF_COEF = 0.5
_C.ANS.PPO.MAX_GRAD_NORM = 0.5
_C.ANS.PPO.TARGET_KL = None
# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
_C.ANS.AGENT = CN()
_C.ANS.AGENT.AGENT_INPUT_SIZE = (84, 84)
_C.ANS.AGENT.AGENT_INPUT_MAPRES = 0.03
_C.ANS.AGENT.SUCCESS_DISTANCE = 0.36
_C.ANS.AGENT.SUBGOAL_SUCCESS_DISTANCE = 0.36
_C.ANS.AGENT.SUBGOAL_REPLAN_STEPS = 8


def get_config(
    config_paths: Optional[Union[List[str], str]] = None, 
    opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config



def finetune_config(
    config,
    seed: int = 0,
    shuffle: bool = False,
    max_scene_repeat_episodes: int = 10,
    width: int = 640,
    height: int = 480
):
    config.defrost()
    config.SEED = seed
    config.ENVIRONMENT.ITERATOR_OPTIONS = CN()
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = shuffle
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = max_scene_repeat_episodes
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = width
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = height
    config.SIMULATOR.RGB_SENSOR.WIDTH = width
    config.SIMULATOR.RGB_SENSOR.HEIGHT = height
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = width
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = height
    config.freeze()
    return config