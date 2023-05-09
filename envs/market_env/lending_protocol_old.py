import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

import gym
from gym import spaces

from utils.agents import AttentionAgent
from envs.market_env.plf_pool import PLFPool
from envs.market_env.market import Market
from envs.market_env.utils import combine_observation_space
from gym.core import RenderFrame, ActType, ObsType
from envs.market_env.constants import (
    CONFIG_LENDING_PROTOCOL,
    CONFIG_PLF_POOL,
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE
)


class LendingProtocol(gym.Env):
    """
    Class implements an AAVE-like over-collateralized lending protocol.
    """

    def __init__(
            self,
            market: Market,
            config: Dict,
    ):
        self.market = market
        self.config = config
        self.plf_pools: List[PLFPool] = list()
        self.agent_mask: List[str] = list()
        self.agent_list = None
        self.observation_space = spaces.Space()
        self.action_space = spaces.Space()  # TODO: implement/initialize the action_space
        self.agent_action_space = list()

        self.reset()

    def reset(self) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        Resets the LendingProtocol and all its plf pools to the initial parameters by reinitializing the PLFPools
        :return:
        """
        self.agent_mask = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.plf_pools = list(
            map(
                lambda params: PLFPool(market=self.market, agent_mask=self.agent_mask, **params),
                self.config[CONFIG_LENDING_PROTOCOL][CONFIG_PLF_POOL]
            )
        )
        self.set_agents(self.agent_list)
        self.observation_space = combine_observation_space(self.plf_pools)
        self.action_space, self.agent_action_space = self._get_action_space(self.plf_pools, self.agent_mask)
        pool_return = [plf_pool.step_result() for plf_pool in self.plf_pools]
        return self._merge_plf_returns(pool_return)

    @staticmethod
    def _initialize_pool(params):
        return PLFPool(**params)

    @staticmethod
    def _get_action_space(plf_pools, agent_mask) -> (spaces.Space, List):
        agent_action_space = [a.n for a in plf_pools[0].action_space]
        compress_action_space = lambda i: spaces.Discrete(sum([pool.action_space[i].n for pool in plf_pools]))
        return spaces.Tuple(list(map(compress_action_space, range(len(agent_mask))))), agent_action_space

    def step(self, action: torch.Tensor) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        :param action: Agent actions are passed in the following format:
                        > [{'gov_action|user_action': int}+]
        :return:
        """
        assert len(self.plf_pools) == len(action), "Number of actions does not match with number of plf pools."
        action_mapped = self._map_action(action)
        pool_return = [plf_pool.step(pool_action) for plf_pool, pool_action in zip(self.plf_pools, action_mapped)]
        return self._merge_plf_returns(pool_return)

    def _map_action(self, action: torch.Tensor) -> List[torch.Tensor]:
        action_mapped = [list() for i in range(len(self.plf_pools))]
        for agent_action, agent_action_space in zip(action, self.agent_action_space):
            plf_idx = int(agent_action) // agent_action_space  # PLF pool chosen by the agent
            plf_action = int(agent_action) % agent_action_space  # Action chosen by the agent
            for i, mapped in enumerate(action_mapped):
                if i == plf_idx:
                    mapped.append(plf_action)
                else:
                    mapped.append(0)  # No action
        # Convert list to tensor
        return [torch.Tensor(a) for a in action_mapped]

    @staticmethod
    def _merge_plf_returns(pool_return: List[Tuple]) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        The returns of the plf_pools have the following form:
        > state
        > reward
        > terminated
        > truncated
        > logs
        :param pool_return:
        :return:
        """
        # 1) Merge observation_spaces/states
        state_list = torch.cat([i[0] for i in pool_return])
        # 2) Reward
        reward_list = torch.cat([i[1] for i in pool_return])  # TODO: The reward have to be summed up!
        # 3) Terminated
        terminated_bool = any([i[2] for i in pool_return])
        # 4) Truncated
        truncated_bool = any([i[3] for i in pool_return])
        # 5) Logs
        logs = {k: v for list_item in pool_return for k, v in list_item[4].items()}
        return state_list, reward_list, terminated_bool, truncated_bool, logs

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def set_agents(self, agent_list: List[AttentionAgent]):
        self.agent_list = agent_list
        list(map(lambda pool: pool.set_agents(agent_list), self.plf_pools))

    def __repr__(self):
        return "LendingProtocol(" + repr(self.plf_pools) + ")"
