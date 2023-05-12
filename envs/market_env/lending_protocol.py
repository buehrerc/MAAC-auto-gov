import logging
import uuid
import pandas as pd
import gym
from gym import spaces
from gym.core import ObsType, RenderFrame
from typing import Dict, List, Tuple, Optional, Union
import torch

from envs.market_env.market import Market
from envs.market_env.plf_pool import PLFPool
from utils.custom_agents import CustomAgent
from envs.market_env.utils import (
    combine_observation_space,
)
from envs.market_env.constants import (
    CONFIG_LENDING_PROTOCOL,
    CONFIG_PLF_POOL,
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    LP_BORROW_SAFETY_MARGIN,
    LP_DEPOSIT_AMOUNT,
)


class InvalidTransaction(Exception):
    def __init__(self, message):
        super().__init__(message)


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

        # Protocol Book-keeping
        self.supply_record: Dict[Tuple, List] = dict()
        self.borrow_record: Dict[Tuple, List] = dict()
        self.reward: torch.Tensor = torch.Tensor()

        # Initialize additional attributes
        self.plf_pools: List[PLFPool] = list()
        self.agent_mask: List[str] = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.agent_list: List[CustomAgent] = list()

        # Gym Environment Attributes
        self.observation_space = spaces.Space()
        self.agent_action_space = list()

        self.reset()

# =====================================================================================================================
#   ENVIRONMENT ACTIONS
# =====================================================================================================================
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
        # Reset the reward
        self.reward = torch.zeros(len(self.agent_mask))
        # Return the state of the pool alongside the remaining return values
        return (
            torch.cat([pool.get_state() for pool in self.plf_pools]),
            self.reward,
            False,
            False,
            dict()
        )

    def step(self, action=None) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        # 1) Update all plf_pools based on the actions of the agents
        pool_states = torch.cat([pool.step() for pool in self.plf_pools])

        # 2) Compute health factors of the loans and get the lowest
        borrow_record_unpacked = [(key[1], key[2], i) for key, item in self.borrow_record.items() for i in item]
        min_health_factor = -torch.inf
        if len(borrow_record_unpacked) > 0:
            min_health_factor = min(list(map(lambda args: self._get_health_factor(*args), borrow_record_unpacked)))

        # 2) Reset the reward
        last_reward = self.reward
        self.reward = torch.zeros(len(self.agent_mask))

        return (
            pool_states,    # Observation State
            last_reward,    # Reward
            False,          # Terminated
            False,          # Truncated
            dict()          # Info
        )

    def _get_health_factor(self, pool_collateral: int, pool_loan: int, loan_hash: str) -> float:
        """
        The health factor of a loan can be computed as follows:

        H = (collateral_factor * loan_amount * loan_price) / (loan_amount * loan_price)
        """
        loan_amount = self.plf_pools[pool_loan].get_borrow(loan_hash)
        loan_price = self.plf_pools[pool_loan].get_token_price()
        collateral_amount = self.plf_pools[pool_collateral].get_supply(loan_hash)
        collateral_price = self.plf_pools[pool_collateral].get_token_price()
        collateral_factor = self.plf_pools[pool_loan].get_collateral_factor()
        return (collateral_amount * collateral_price * collateral_factor) / (loan_amount * loan_price)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def set_agent(self, agent_list: List[CustomAgent]):
        self.agent_list = agent_list

# =====================================================================================================================
#   AGENT ACTIONS
# =====================================================================================================================
    def update_collateral_factor(self, pool_id: int, direction: int) -> (float, bool):
        return self.plf_pools[pool_id].update_collateral_factor(direction)

    def deposit(self, agent_id: int, pool_to: int, amount: float) -> (float, bool):
        # 1) Deduct the funds from the agents balance
        reward, success = self._remove_agent_funds(agent_id, pool_to, amount)
        if not success:  # Agent doesn't have enough funds
            return reward, success

        # 2) Record the deposit
        deposit_hash = uuid.uuid4().hex
        if self.supply_record.get((agent_id, pool_to)) is None:
            self.supply_record[agent_id, pool_to] = list()
        self.supply_record[agent_id, pool_to].append(deposit_hash)

        # 3) Add the funds to the pool
        self.plf_pools[pool_to].add_supply(key=deposit_hash, amount=amount)

        return 0, True

    def withdraw(self, agent_id: int, pool_from: int, amount: float) -> (float, bool):
        pool_token = self.plf_pools[pool_from].get_token_name()

        # 1) Check whether the user has deposited anything into the pool
        if self.supply_record.get((agent_id, pool_from)) is None or len(self.supply_record.get((agent_id, pool_from))) == 0:
            logging.info(
                f"Agent {agent_id} tried to withdraw {amount} from {pool_token}, "
                f"but didn't deposit enough or didn't deposit at all."
            )
            # TODO: Reward -> negative reward
            return 0, False

        # 2) Get the hash value of the supply
        withdraw_hash = self.supply_record.get((agent_id, pool_from)).pop()

        # 3) Remove the funds from the pool
        withdraw_amount = self.plf_pools[pool_from].remove_supply(withdraw_hash)

        # 4) Add the funds to the agents balance
        self.agent_list[agent_id].add_balance(token_name=pool_token, amount=withdraw_amount)

        return 0.0, True

    def borrow(self, agent_id: int, pool_collateral: int, pool_loan: int, amount: float) -> (float, bool):
        """
        The agent borrows funds with a safety margin which is set as a constant (LP_BORROW_SAFETY_MARGIN)
        borrow_amount = (deposit_amount * deposit_price * (borrow_collateral_factor + LP_BORROW_SAFETY_MARGIN)) / borrow_price

        :param agent_id: ID of agent who tries to borrow
        :param pool_collateral: PLFPool, to which the collateral will be deposited
        :param pool_loan: PLFPool, from which the funds are borrowed
        :amount: Amount of funds that are being deposited
        """
        # 1) Compute the deposit and borrow amount
        deposit_amount = LP_DEPOSIT_AMOUNT
        deposit_price = self.plf_pools[pool_collateral].get_token_price()
        borrow_price = self.plf_pools[pool_loan].get_token_price()
        L_t = self.plf_pools[pool_loan].get_collateral_factor() - LP_BORROW_SAFETY_MARGIN
        borrow_amount = (deposit_amount * deposit_price * L_t) / borrow_price

        # 2) Deduct the collateral from the agent first
        reward, success = self._remove_agent_funds(agent_id, pool_collateral, amount)
        # Agent doesn't have enough funds
        if not success:
            logging.info(
                f"Agent {agent_id} tried to borrow funds from pool {pool_loan} by "
                f"providing collateral to pool {pool_collateral}, but didn't have enough funds."
            )
            return reward, success

        # 3) Record the borrow
        loan_hash = uuid.uuid4().hex
        if self.borrow_record.get((agent_id, pool_collateral, pool_loan)) is None:
            self.borrow_record[agent_id, pool_collateral, pool_loan] = list()
        self.borrow_record[agent_id, pool_collateral, pool_loan].append(loan_hash)

        # 4) Deposit the collateral (2nd transaction for step 2)
        self.plf_pools[pool_collateral].add_supply(key=loan_hash, amount=deposit_amount)

        # 4) Withdraw the borrowed funds
        borrow_token = self.plf_pools[pool_loan].get_token_name()
        self.agent_list[agent_id].add_balance(token_name=borrow_token, amount=borrow_amount)
        self.plf_pools[pool_loan].start_borrow(key=loan_hash, amount=borrow_amount)

        return 0, True

    def repay(self, agent_id: int, pool_loan: int, pool_collateral: int, amount: float) -> (float, bool):
        """
        :param agent_id: ID of agent who tries to repay the loan
        :param pool_loan: PLFPool, to which loan is repaid
        :param pool_collateral: PLFPool, from which the collateral is repaid
        :amount: Amount of funds that are being deposited
        """
        # 1) Check whether an according borrow exists
        if self.borrow_record.get((agent_id, pool_collateral, pool_loan)) is None or \
                len(self.borrow_record.get((agent_id, pool_collateral, pool_loan))) == 0:
            logging.info(
                f"Agent {agent_id} tried to repay a loan, but never borrowed funds from pool {pool_loan}."
            )
            # TODO: Reward -> negative reward
            return 0, False

        # 2) Retrieve the loan hash
        loan_hash = self.borrow_record[(agent_id, pool_collateral, pool_loan)][0]

        # 3) Agent pays the borrowed funds
        borrowed_amount = self.plf_pools[pool_loan].return_borrow(loan_hash)
        reward, success = self._remove_agent_funds(agent_id, pool_loan, borrowed_amount)
        if not success:
            # Agent cannot repay the borrowed funds -> reset the borrowed funds in the pool
            self.plf_pools[pool_loan].get_borrow(loan_hash, borrowed_amount)
            logging.info(
                f"Agent {agent_id} tried to repay the loan from pool {pool_loan}, "
                f"but didn't have enough funds."
            )
            return 0, False

        # 4) Transfer the collateral back to the agent
        collateral_token = self.plf_pools[pool_collateral].get_token_name()
        collateral_amount = self.plf_pools[pool_collateral].remove_supply(loan_hash)
        self.agent_list[agent_id].add_balance(token_name=collateral_token, amount=collateral_amount)

        # 5) Remove the borrow record
        self.borrow_record[(agent_id, pool_collateral, pool_loan)].pop(0)
        return 0, True

    def liquidate(self, agent_id: int, pool_id: int) -> (float, bool):
        pass

    def _remove_agent_funds(self, agent_id: int, pool_to: int, amount: float):
        pool_token = self.plf_pools[pool_to].get_token_name()
        agent_balance = self.agent_list[agent_id].get_balance(pool_token)
        if agent_balance < amount:
            logging.info(
                f"Agent {agent_id} tried to deposit {amount} into {pool_token},"
                f" but didn't have enough funds ({agent_balance})"
            )
            # TODO: Reward -> negative reward
            return 0, False
        self.agent_list[agent_id].sub_balance(token_name=pool_token, amount=amount)
        return 0, True

    def __repr__(self):
        return "LendingProtocol(" + repr(self.plf_pools) + ")"
