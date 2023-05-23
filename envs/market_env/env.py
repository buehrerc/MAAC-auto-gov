import gym
import torch
import logging
from typing import Tuple, Optional, Union, List, Dict
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame
from utils.rewards import reward_function
from envs.market_env.lending_protocol import LendingProtocol
from envs.market_env.market import Market
from envs.market_env.utils import combine_observation_space, encode_action
from envs.market_env.constants import (
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER,
    CONFIG_AGENT_BALANCE,
    CONFIG_AGENT_REWARD,
    PLF_STEP_SIZE,
    AGENT_OBSERVATION_SPACE,
)


class MultiAgentEnv(gym.Env):
    def __init__(
        self,
        config: Dict,
        seed: int,
    ) -> None:
        """
        :param config: Environment Configs
        """
        super(MultiAgentEnv).__init__()

        self.config = config

        # Agent Parameters
        self.agent_mask: List[str] = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.agent_reward: List[str] = [agent[CONFIG_AGENT_REWARD] for agent in self.config[CONFIG_AGENT]]

        # Environment Attributes
        self.market = Market(config=self.config, seed=seed)
        self.lending_protocol = LendingProtocol(owner=self.agent_mask.index(CONFIG_AGENT_TYPE_GOVERNANCE),
                                                market=self.market, config=self.config)

        # Initialize agent balances
        self.agent_balance: List[Dict] = self._initialize_agent_balance()

        # Interact with lending_protocol
        agent_observation_space = [AGENT_OBSERVATION_SPACE(len(self.lending_protocol.plf_pools)) for _ in self.agent_mask]

        # Gym Attributes
        self.observation_space = combine_observation_space([self.lending_protocol, self.market],
                                                           obs_spaces=agent_observation_space)
        self.action_encoding = encode_action(len(self.lending_protocol.plf_pools))
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_encoding[agent_type]))
                                          for agent_type in self.agent_mask])

    def reset(self) -> torch.Tensor:
        # Reset the internal state
        lp_state = self.lending_protocol.reset()
        market_state = self.market.reset()
        self.agent_balance = self._initialize_agent_balance()

        # Append the agents state
        agent_state = self._get_agent_state()
        state = torch.cat([lp_state, market_state, agent_state])
        return state

    def _initialize_agent_balance(self) -> List[Dict]:
        """
        Function generates a dictionary for each agent.
        Within each dictionary, all tokens of the market have a balance.
        """
        agent_balance = [
            {
                token_name: agent_dict.get(CONFIG_AGENT_BALANCE, {}).get(token_name, 0)
                for token_name in self.market.tokens.keys()
            }
            for agent_dict in self.config[CONFIG_AGENT]
        ]
        self.lending_protocol.set_agent_balance(agent_balance)
        return agent_balance

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: ActType) -> Tuple[ObsType, torch.Tensor, List[bool], dict]:
        """
        Step first applies actions of the agents to the LendingPool.
        Afterwards, the Market gets updated.
        Finally, the rewards of the agents are collected.

        :param action: Action of the users

        :return: new observations, rewards, done, logs
        """
        # 1) Set the action of each agent first:
        action_feedback = self._set_action(action)

        # 2) Update the whole environment based on the actions of the agents
        lp_state = self.lending_protocol.update()
        market_state = self.market.update()
        agent_state = self._get_agent_state()
        state = torch.cat([lp_state, market_state, agent_state])

        # 3) Collect the rewards for all
        reward = torch.Tensor([
            reward_function(i, args[0], self.lending_protocol, args[1])
            for i, args in enumerate(zip(self.agent_reward, action_feedback))
        ])

        return (
            state,                           # Observation State
            reward,                          # Reward
            [False] * len(self.agent_mask),  # Terminated
            dict()                           # Info
        )

    def _get_agent_state(self) -> torch.Tensor:
        agent_state = sum([list(balance.values()) for balance in self.agent_balance], [])
        return torch.Tensor(agent_state)

    def _set_action(self, action_list: torch.Tensor) -> List:
        """
        :return: List of feedback for each action (True: illegal_action, False: legal_action)
        """
        action_feedback = list()
        for agent_id, action in enumerate(action_list):
            if self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_GOVERNANCE:
                action_feedback.append(self._set_governance_action(agent_id, action))
            elif self.agent_mask[agent_id] == CONFIG_AGENT_TYPE_USER:
                action_feedback.append(self._set_user_action(agent_id, action))
            else:
                raise KeyError("Agent type {} was not found".format(self.agent_mask[agent_id]))
        return action_feedback

    def _set_governance_action(self, agent_id: int, action: torch.Tensor) -> bool:
        """
        A governance agent can pick between the following actions:
            > No action
            > lower collateral factor of pool i
            > raise collateral factor of pool i

        :param agent_id: ID of agent who wants to perform an action
        :param action: Encoded action of the agent

        :return: True: illegal_action, False: legal_action
        """
        action_id, pool_id = self.action_encoding[CONFIG_AGENT_TYPE_GOVERNANCE][int(action)]
        if action_id == 0:  # No Action
            logging.debug("Agent {}: No action".format(agent_id, pool_id))
            return False
        elif action_id == 1:  # Lower Collateral Factor
            logging.debug("Agent {}: Lower collateral factor of pool {}".format(agent_id, pool_id))
            return self.lending_protocol.update_collateral_factor(pool_id, -1)
        elif action_id == 2:  # Raise Collteral Factor
            logging.debug("Agent {}: Raise collateral factor of pool {}".format(agent_id, pool_id))
            return self.lending_protocol.update_collateral_factor(pool_id, 1)
        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))

    def _set_user_action(self, agent_id: int, action: torch.Tensor) -> bool:
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

        :return: True: illegal_action, False: legal_action
        """
        action_id, idx_from, idx_to = self.action_encoding[CONFIG_AGENT_TYPE_USER][int(action)]

        if action_id == 0:  # no action
            logging.debug("Agent {}: No action".format(agent_id))
            return False

        elif action_id == 1:  # deposit
            logging.debug("Agent {}: Deposit {} funds into pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            return self.lending_protocol.deposit(agent_id, idx_to, PLF_STEP_SIZE)

        elif action_id == 2:  # withdraw
            logging.debug("Agent {}: Withdraw {} funds in pool {}".format(agent_id, PLF_STEP_SIZE, idx_to))
            return self.lending_protocol.withdraw(agent_id, idx_to, PLF_STEP_SIZE)

        elif action_id == 3:  # borrow
            logging.debug("Agent {}: Borrow funds from {} & collateral to {}".format(agent_id, idx_from, idx_to))
            return self.lending_protocol.borrow(agent_id, idx_to, idx_from, PLF_STEP_SIZE)

        elif action_id == 4:  # repay
            logging.debug("Agent {}: Repay funds to {} & receive collateral form {}".format(agent_id, idx_to, idx_from))
            return self.lending_protocol.repay(agent_id, idx_to, idx_from, PLF_STEP_SIZE)

        elif action_id == 5:  # liquidate
            logging.debug("Agent {}: Liquidate pool {}".format(agent_id, idx_to))
            return self.lending_protocol.liquidate(agent_id, idx_to)

        else:
            raise NotImplementedError("Action Code {} is unknown".format(action_id))
