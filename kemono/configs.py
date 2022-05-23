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