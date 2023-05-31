from typing import List, Dict, Tuple
import numpy as np
from gym import spaces

from envs.market_env.constants import (
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER,
    ENVIRONMENT_STATES,
    CONFIG_MARKET,
    CONFIG_TOKEN,
    CONFIG_LENDING_PROTOCOL,
    CONFIG_PLF_POOL,
    CONFIG_AGENT
)


def combine_observation_space(obj_list: List[object], obs_spaces: List[spaces.Space] = None):
    observation_spaces = [o.observation_space for o in obj_list]
    if obs_spaces is not None and len(obs_spaces) > 0:
        observation_spaces.extend(obs_spaces)
    return spaces.Box(
        low=np.concatenate([o.low for o in observation_spaces]),
        high=np.concatenate([o.high for o in observation_spaces])
    )


def generate_state_mapping(env_config) -> List[str]:
    token_num = len(env_config[CONFIG_MARKET][CONFIG_TOKEN])
    plf_num = [len(lending_protocol[CONFIG_PLF_POOL]) for lending_protocol in env_config[CONFIG_LENDING_PROTOCOL]]
    agent_num = len(env_config[CONFIG_AGENT])
    return ENVIRONMENT_STATES(agent_num, token_num, plf_num)


def encode_action(num_plf_pools: List[int]) -> (spaces.Space, Dict):
    encoding = {
        CONFIG_AGENT_TYPE_GOVERNANCE: encode_governance_action(num_plf_pools),
        CONFIG_AGENT_TYPE_USER: encode_user_action(num_plf_pools)
    }
    return encoding


def encode_governance_action(num_plf_pools: int) -> Dict[int, Tuple]:
    """
    ENCODING CONVENTIONS:
    > GOVERNANCE AGENT:
        0: no action
        1: lower collateral factor in 1st pool
        2: raise collateral factor in 1st pool
        3: lower borrow slope_1 in 1st pool
        4: raise borrow slope_1 in 1st pool
        5: lower borrow slope_2 in 1st pool
        6: raise borrow slope_2 in 1st pool
        7: lower collateral factor in 2nd pool
        8: raise collateral factor in 2nd pool
        etc.

    > ACTION SPACE: 1 + num_plf_pool * 6
    """
    encoding = dict()
    i = 1
    encoding[0] = (0, None)
    for n in range(num_plf_pools):
        encoding[i] = (1, n)
        encoding[i+1] = (2, n)
        encoding[i+2] = (3, n)
        encoding[i+3] = (4, n)
        encoding[i+4] = (5, n)
        encoding[i+5] = (6, n)
        i += 6
    return encoding


def encode_user_action(plf_pool_in_lp: List[int]) -> Dict[int, Tuple]:
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
    encoding[0] = (0, None, None, None)
    for i_lp, num_plf_pools in enumerate(plf_pool_in_lp):
        for n in range(num_plf_pools):
            encoding[i] = (1, i_lp, None, n)      # Deposit
            encoding[i+1] = (2, i_lp, None, n)    # Withdraw
            encoding[i+2] = (5, i_lp, None, n)    # Liquidate
            i += 3
            for m in range(num_plf_pools):
                if m == n:
                    continue
                encoding[i] = (3, i_lp, n, m)     # Borrow
                encoding[i+1] = (4, i_lp, n, m)   # Repay
                i += 2
    return encoding
