import logging
import gym
from gym import spaces
from gym.core import ObsType, RenderFrame
from typing import Dict, List, Tuple, Optional, Union
import torch

from envs.market_env.market import Market
from envs.market_env.plf_pool import PLFPool
from utils.agents import AttentionAgent
from envs.market_env.utils import (
    combine_observation_space,
    encode_action
)
from envs.market_env.constants import (
    CONFIG_LENDING_PROTOCOL,
    CONFIG_PLF_POOL,
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER, PLF_STEP_SIZE,
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

        # Initialize additional attributes
        self.plf_pools: List[PLFPool] = list()
        self.agent_mask: List[str] = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.agent_list = None

        # Reward Parameters
        self.reward: torch.Tensor = torch.Tensor()

        # Gym Environment Attributes
        self.observation_space = spaces.Space()
        self.action_space = spaces.Space()
        self.agent_action_space = list()
        self.action_encoding = dict()

        self.reset()

    def reset(self) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        """
        Resets the LendingProtocol and all its plf pools to the initial parameters by reinitializing the PLFPools
        :return: Tuple[ObsType, torch.Tensor, bool, bool, dict]
        """
        # (Re-)Initialize the plf pools
        self.plf_pools = list(
            map(
                lambda params: PLFPool(market=self.market, **params),
                self.config[CONFIG_LENDING_PROTOCOL][CONFIG_PLF_POOL]
            )
        )
        # Based on the reinitialized plf pools -> compute observation and action space
        self.observation_space = combine_observation_space(self.plf_pools)
        self.action_encoding = encode_action(len(self.plf_pools))
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_encoding[agent_type])) for agent_type in self.agent_mask])
        # Reset the reward
        self.reward = torch.empty(len(self.agent_mask))
        # Return the state of the pool alongside the remaining return values
        return (
            torch.cat([pool.get_state() for pool in self.plf_pools]),
            self.reward,
            False,
            False,
            dict()
        )

    def step(self, action_list: torch.Tensor) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        # 1) Let all agents perform their action
        for agent_id, action in enumerate(action_list):
            if self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_GOVERNANCE:
                self.step_governance(agent_id, action)
            elif self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_USER:
                self.step_user(agent_id, action)
            else:
                raise KeyError("Agent type {} was not found".format(self.agent_mask[agent_id]))
        # 2) Update all plf_pools based on the actions of the agents
        pool_states = torch.cat([pool.step() for pool in self.plf_pools])
        # 3) Reset the reward
        last_reward = self.reward
        self.reward = torch.empty(len(self.agent_mask))
        return (
            pool_states,    # Observation State
            last_reward,    # Reward
            False,          # Terminated
            False,          # Truncated
            dict()          # Info
        )

    def step_governance(self, agent_id: int, action: torch.Tensor) -> None:
        action_id, pool_id = self.action_encoding[CONFIG_AGENT_TYPE_GOVERNANCE][int(action)]
        if action_id == 0:  # No Action
            logging.info("Agent {}: No action".format(agent_id, pool_id))
        elif action_id == 1:  # Lower Collateral Factor
            self.plf_pools[pool_id].update_collateral_factor(-1)
            logging.info("Agent {}: Lower collateral factor of pool {}".format(agent_id, pool_id))
        elif action_id == 2:  # Raise Collteral Factor
            self.plf_pools[pool_id].update_collateral_factor(1)
            logging.info("Agent {}: Raise collateral factor of pool {}".format(agent_id, pool_id))
        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))

    def step_user(self, agent_id: int, action: torch.Tensor) -> None:
        action_id, idx_from, idx_to = self.action_encoding[CONFIG_AGENT_TYPE_USER][int(action)]
        if action_id == 0:  # no action
            logging.info("Agent {}: No action".format(agent_id))
            pass
        elif action_id == 1:  # deposit
            logging.info("Agent {}: Deposit {} funds into pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            self.plf_pools[idx_to].do_deposit(agent_id, self.agent_list[agent_id], PLF_STEP_SIZE)
        elif action_id == 2:  # withdraw
            logging.info("Agent {}: Withdraw {} funds in pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            self.plf_pools[idx_to].do_withdraw(agent_id, self.agent_list[agent_id], PLF_STEP_SIZE)
        elif action_id == 3:  # borrow
            logging.info("Agent {}: Borrow funds: {} => {}".format(agent_id, idx_from, idx_to))
            # TODO: compute amount that can be borrowed (using the collateral factor)
        elif action_id == 4:  # repay
            logging.info("Agent {}: Repay funds: {} => {}".format(agent_id, idx_from, idx_to))
            # TODO:
        elif action_id == 5:  # liquidate
            logging.info("Agent {}: Liquidate funds: {} => {}".format(agent_id, idx_from, idx_to))
            # TODO:
        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))

    def set_agents(self, agent_list: List[AttentionAgent]) -> None:
        self.agent_list = agent_list

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def __repr__(self):
        return "LendingProtocol(" + repr(self.plf_pools) + ")"
