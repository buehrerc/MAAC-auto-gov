from typing import List, Dict, Tuple
import numpy as np
import gym
from gym import spaces

from envs.market_env.constants import (
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER,
)


def combine_observation_space(obj_list: List[gym.Env], obs_spaces: List[spaces.Space] = None):
    observation_spaces = [o.observation_space for o in obj_list]
    if obs_spaces is not None and len(obs_spaces) > 0:
        observation_spaces.extend(obs_spaces)
    return spaces.Box(
        low=np.concatenate([o.low for o in observation_spaces]),
        high=np.concatenate([o.high for o in observation_spaces])
    )


def encode_action(num_plf_pools) -> (spaces.Space, Dict):
    encoding = {
        CONFIG_AGENT_TYPE_GOVERNANCE: encode_governance_action(num_plf_pools),
        CONFIG_AGENT_TYPE_USER: encode_user_action(num_plf_pools)
    }
    return encoding


def encode_governance_action(num_plf_pols: int) -> Dict[int, Tuple]:
    """
    ENCODING CONVENTIONS:
    > GOVERNANCE AGENT:
        0: no action
        1: lower collateral factor in 1st pool
        2: raise collateral factor in 1st pool
        3: lower collateral factor in 2nd pool
        4: raise collateral factor in 2nd pool
        etc.
    """
    encoding = dict()
    i = 1
    encoding[0] = (0, None)
    for n in range(num_plf_pols):
        encoding[i] = (1, n)
        encoding[i+1] = (2, n)
        i += 2
    return encoding


def encode_user_action(num_plf_pools: int) -> Dict[int, Tuple]:
    """
    ENCODING CONVENTIONS:
    > USER AGENT:
        0: no action
        1: deposit funds into 1st pool
        2: withdraw funds from 1st pool
        3: liquidate 1st pool
        4: borrow funds from 1st pool and deposit funds in 2nd pool
        5: borrow funds from 1st pool and deposit funds in 3rd pool
        etc.
        i: repay funds to 1st pool and receive collateral from 2nd pool
        i+1: repay funds to 1st pool and receive collateral from 3rd pool
        etc.

    > ACTION SPACE: 1 + num_plf_pools * (3 + 2 * (num_plf_pools - 1))

    """
    encoding = dict()
    i = 1
    encoding[0] = (0, None, None)
    for n in range(num_plf_pools):
        encoding[i] = (1, None, n)      # Deposit
        encoding[i+1] = (2, None, n)    # Withdraw
        encoding[i+2] = (5, None, n)    # Liquidate
        i += 3
        for m in range(num_plf_pools):
            if m == n:
                continue
            encoding[i] = (3, n, m)     # Borrow
            encoding[i+1] = (4, n, m)   # Repay
            i += 2
    return encoding
