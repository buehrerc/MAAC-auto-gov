import logging
from typing import Tuple, Optional, Union, List, Dict

import gym
import torch
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame

from envs.market_env.lending_protocol import LendingProtocol
from envs.market_env.market import Market
from envs.market_env.utils import combine_observation_space, encode_action
from envs.market_env.constants import (
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER,
    CONFIG_AGENT_BALANCE,
    PLF_STEP_SIZE,
    AGENT_OBSERVATION_SPACE,
)


class MultiAgentEnv(gym.Env):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        """
        :param config: Environment Configs
        """
        super(MultiAgentEnv).__init__()

        self.config = config
        self.agent_mask: List[str] = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.market = Market(config=self.config)
        gov_agent_id = self.agent_mask.index(CONFIG_AGENT_TYPE_GOVERNANCE)
        self.lending_protocol = LendingProtocol(owner=gov_agent_id, market=self.market, config=self.config)

        # Agent balances
        self.agent_balance: List[Dict] = [
            {
                token_name: agent_dict.get(CONFIG_AGENT_BALANCE, {}).get(token_name, 0)
                for token_name in self.market.tokens.keys()
            }
            for agent_dict in self.config[CONFIG_AGENT]
        ]

        # Interact with lending_protocol
        agent_observation_space = [AGENT_OBSERVATION_SPACE(len(self.lending_protocol.plf_pools)) for _ in self.agent_mask]
        self.lending_protocol.set_agent_balance(self.agent_balance)

        # Gym Attributes
        self.observation_space = combine_observation_space([self.lending_protocol, self.market], obs_spaces=agent_observation_space)
        self.action_encoding = encode_action(len(self.lending_protocol.plf_pools))
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_encoding[agent_type])) for agent_type in self.agent_mask])

    def reset(self) -> torch.Tensor:
        lp_state = self.lending_protocol.reset()
        market_state = self.market.reset()
        # Append the agents state
        agent_state = self._get_agent_state()
        state = torch.cat([lp_state, market_state, agent_state])
        return state

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: ActType) -> Tuple[ObsType, torch.Tensor, List[bool], dict]:
        """
        Step first applies actions of the agents to the LendingPool. Afterwards, the Market gets updated.
        :param action: Action of the users
        :return: new observations, rewards, done, logs
        """
        # Set the action of each agent first:
        initial_reward = self._set_action(action)

        # Update the whole environment based on the actions of the agents
        lp_state, reward, terminated, lp_logs = self.lending_protocol.update()
        market_state, _, _, market_logs = self.market.update()
        agent_state = self._get_agent_state()
        state = torch.cat([lp_state, market_state, agent_state])

        return state, reward, terminated, lp_logs

    def _get_agent_state(self) -> torch.Tensor:
        agent_state = sum([list(balance.values()) for balance in self.agent_balance], [])
        return torch.Tensor(agent_state)

    def _set_action(self, action_list: torch.Tensor) -> List:
        action_returns = list()
        for agent_id, action in enumerate(action_list):
            if self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_GOVERNANCE:
                action_returns.append(self._set_governance_action(agent_id, action))
            elif self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_USER:
                action_returns.append(self._set_user_action(agent_id, action))
            else:
                raise KeyError("Agent type {} was not found".format(self.agent_mask[agent_id]))
        return action_returns

    def _set_governance_action(self, agent_id: int, action: torch.Tensor) -> (float, bool):
        """
        A governance agent can pick between the following actions:
            > No action
            > lower collateral factor of pool i
            > raise collateral factor of pool i

        :param agent_id: ID of agent who wants to perform an action
        :param action: Encoded action of the agent
        :return: (reward, success)
        """
        action_id, pool_id = self.action_encoding[CONFIG_AGENT_TYPE_GOVERNANCE][int(action)]
        if action_id == 0:  # No Action
            logging.info("Agent {}: No action".format(agent_id, pool_id))
            return 0.0, True
        elif action_id == 1:  # Lower Collateral Factor
            logging.info("Agent {}: Lower collateral factor of pool {}".format(agent_id, pool_id))
            return self.lending_protocol.update_collateral_factor(pool_id, -1)
        elif action_id == 2:  # Raise Collteral Factor
            logging.info("Agent {}: Raise collateral factor of pool {}".format(agent_id, pool_id))
            return self.lending_protocol.update_collateral_factor(pool_id, 1)
        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))

    def _set_user_action(self, agent_id: int, action: torch.Tensor) -> (float, bool):
        """
        A governance agent can pick between the following actions:
            > No action
            > Deposit funds into pool i
            > Withdraw funds from pool i
            > Borrow funds from pool i by providing collateral to pool j
            > Repay loan to pool i and receive collateral from pool j
            > Liquidate pool i

        :param agent_id: ID of agent who wants to perform an action
        :param action: Encoded action of the agent
        :return: (reward, success)
        """
        action_id, idx_from, idx_to = self.action_encoding[CONFIG_AGENT_TYPE_USER][int(action)]

        if action_id == 0:  # no action
            logging.info("Agent {}: No action".format(agent_id))
            return 0.0, True

        elif action_id == 1:  # deposit
            logging.info("Agent {}: Deposit {} funds into pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            return self.lending_protocol.deposit(agent_id, idx_to, PLF_STEP_SIZE)

        elif action_id == 2:  # withdraw
            logging.info("Agent {}: Withdraw {} funds in pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            return self.lending_protocol.withdraw(agent_id, idx_to, PLF_STEP_SIZE)

        elif action_id == 3:  # borrow
            logging.info("Agent {}: Borrow funds from {} & collateral to {}".format(agent_id, idx_from, idx_to))
            return self.lending_protocol.borrow(agent_id, idx_to, idx_from, PLF_STEP_SIZE)

        elif action_id == 4:  # repay
            logging.info("Agent {}: Repay funds to {} & receive collateral form {}".format(agent_id, idx_to, idx_from))
            return self.lending_protocol.repay(agent_id, idx_to, idx_from, PLF_STEP_SIZE)

        elif action_id == 5:  # liquidate
            logging.info("Agent {}: Liquidate pool {}".format(agent_id, idx_to))
            return self.lending_protocol.liquidate(agent_id, idx_to)

        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))
