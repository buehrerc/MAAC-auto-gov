from typing import Dict, List, Tuple, Optional, Union

import torch
import gym
from gym import spaces
from gym.core import ObsType, RenderFrame

from envs.market_env.market import Market
from envs.market_env.plf_pool import PLFPool


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

        # Initialize additional attributes
        self.plf_pools: List[PLFPool] = list()
        self.agent_mask: List[str] = list()
        self.agent_list: None

        # Gym Environment Attributes
        self.observation_space = spaces.Space()
        self.action_space = spaces.Space()
        self.agent_action_space = list()

        self.reset()

    def reset(self) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        Resets the LendingProtocol and all its plf pools to the initial parameters by reinitializing the PLFPools
        :return: Tuple[ObsType, torch.Tensor, bool, bool, dict]
        """
        pass

    def step(self, action_list: torch.Tensor) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        agent_results = list()
        for agent_id, action in enumerate(action_list):
            action_id, idx_from, idx_to = self._map_action(action)
            if action_id == 0:  # no action
                pass
            elif action_id == 1:  # deposit
                self.plf_pools[idx_to].do_deposit(agent_id)
            elif action_id == 2:  # withdraw
                self.plf_pools[idx_to].do_withdraw(agent_id)
            elif action_id == 3:  # borrow
                # TODO: access amount that can be borrowed (using the collateral factor)
                raise NotImplementedError()
            elif action_id == 4:  # repay
                raise NotImplementedError()
            elif action_id == 5:  # liquidate
                raise NotImplementedError()

        return self._merge_results(agent_results)

    def _map_action(self, action: torch.Tensor) -> (int, int, int):
        # TODO: Find an intelligent way to encode the options
        return 0,0,0

    def _merge_results(self, results: List) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass