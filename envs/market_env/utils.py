from typing import List
import numpy as np
import gym
from gym import spaces


def combine_observation_space(obj_list: List):
    observation_spaces = [o.observation_space for o in obj_list]
    return spaces.Box(
        low=np.concatenate([o.low for o in observation_spaces]),
        high=np.concatenate([o.high for o in observation_spaces])
    )
