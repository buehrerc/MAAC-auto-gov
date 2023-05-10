import logging
from typing import Tuple, Optional, Union, List, Dict

import gym
import torch
from gym.core import ActType, ObsType, RenderFrame
from envs.market_env.lending_protocol import LendingProtocol
from envs.market_env.market import Market
from envs.market_env.utils import combine_observation_space
from utils.agents import AttentionAgent


class MultiAgentEnv(gym.Env):
    def __init__(
        self,
        config: Dict,
        market: Market,
        lending_protocol: LendingProtocol,
    ) -> None:
        """
        :param config: Environment Configs
        :param market: Market object holding initialized Tokens
        :param lending_protocol: LendingProtocol holding initialized PLFPools
        """
        super(MultiAgentEnv).__init__()
        self.config = config
        self.lending_protocol = lending_protocol
        self.market = market
        self.observation_space = combine_observation_space([lending_protocol, market])
        self.action_space = lending_protocol.action_space

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: ActType) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        Step first applies actions of the agents to the LendingPool. Afterwards, the Market gets updated.
        :param action: Action of the users
        :return:
        """
        lp_state, reward, terminated, truncated, lp_logs = self.lending_protocol.step(action)
        market_state, _, _, _, market_logs = self.market.step(None)
        state = torch.cat([lp_state, market_state])
        return state, reward, terminated, truncated, lp_logs

    def reset(self) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        lp_state, reward, terminated, truncated, lp_logs = self.lending_protocol.reset()
        market_state, _, _, _, market_logs = self.market.reset()
        state = torch.cat([lp_state, market_state])
        return state, reward, terminated, truncated, lp_logs

    def set_agents(self, agent_list: List[AttentionAgent]) -> None:
        self.lending_protocol.set_agents(agent_list)
