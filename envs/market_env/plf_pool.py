import logging
import torch
import numpy as np
from gym.core import ObsType
from typing import Dict, List
from utils.custom_agents import UserAgent
from envs.market_env.market import Market
from envs.market_env.constants import (
    INITIATOR,
    COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_RB_FACTOR,
    PLF_SPREAD,
    PLF_OBSERVATION_SPACE,
    PLF_ACTION_SPACE
)


class PLFPool:
    def __init__(
        self,
        market: Market,
        token_name: str = "dai",
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        col_factor_change_rate: float = COLLATERAL_FACTOR_CHANGE_RATE,
        rb_factor: float = PLF_RB_FACTOR,
        spread: float = PLF_SPREAD,
        seed: int = 0
    ):
        assert 0 < collateral_factor < 1, "Collateral Factor must be between 0 and 1"
        # Gym Atributes
        self.observation_space = PLF_OBSERVATION_SPACE
        self.action_space = PLF_ACTION_SPACE

        # General Properties
        self.token_name = token_name
        self.token = market.get_token(self.token_name)
        self.agent_list = None

        # Pool Parameters
        self.collateral_factor: float = collateral_factor
        self.col_factor_change_rate: float = col_factor_change_rate
        self.supply_token: Dict[str, List] = {INITIATOR: [initial_starting_funds]}
        self.borrow_token: Dict[str, List] = dict()

        # Reward Parameters
        self.previous_reserve: float = 0.0

        self.rb_factor = rb_factor
        self.spread = spread

        # Randomness
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def total_supply_token(self) -> float:
        return sum([sum(v) for v in self.supply_token.values()])

    @property
    def total_borrow_token(self) -> float:
        return sum([sum(v) for v in self.borrow_token.values()])

    @property
    def utilization_ratio(self) -> float:
        if self.total_supply_token == 0:
            return 0
        return self.total_borrow_token / self.total_supply_token

    @property
    def supply_interest_rate(self) -> float:
        assert self.utilization_ratio > -1e-9, "Utilization Ratio must be non-negative"
        if self.utilization_ratio == 0:
            return 0.0
        constrained_util_ratio = max(0, min(self.utilization_ratio, 0.97))
        daily_borrow_interest = (1 + self.borrow_interest_rate) ** (1 / 365) - 1
        daily_supply_interest = daily_borrow_interest * constrained_util_ratio
        return ((1 + daily_supply_interest) ** 365 - 1) * (1 - self.spread)

    @property
    def borrow_interest_rate(self) -> float:
        assert self.utilization_ratio > -1e-9, "Utilization Ratio must be non-negative"
        if self.utilization_ratio == 0:
            return 0.0
        constrained_util_ratio = max(0, min(self.utilization_ratio, 0.97))
        return constrained_util_ratio / (self.rb_factor * (1 - constrained_util_ratio))

    @property
    def reserve(self) -> float:
        return self.total_supply_token - self.total_borrow_token

    @staticmethod
    def _get_daily_interest(interest: float) -> float:
        return (1 + interest) ** (1 / 365)

    def get_state(self) -> torch.Tensor:
        return torch.Tensor([
            self.total_supply_token,    # Supply Token
            self.total_borrow_token,    # Borrow Token
            self.reserve,               # Net Position
            self.utilization_ratio,     # Utilization Ratio
            self.collateral_factor,     # Collateral Factor
            self.supply_interest_rate,  # Supply Interest Rate
            self.borrow_interest_rate   # Borrow Interest Rate
        ])

    def step(self) -> ObsType:
        """
        Function is being used to update the pool internal parameters
        """
        self.accrue_interest()
        self.previous_reserve = self.reserve
        return self.get_state()

    def accrue_interest(self) -> None:
        """
        Accrue interest to all users in the pool
        :return: None
        """
        # Accrue interest tokens
        for user_name, token_amount in self.supply_token.items():
            for i in range(len(self.supply_token[user_name])):
                self.supply_token[user_name][i] *= self._get_daily_interest(self.supply_interest_rate)
        # Accrue borrow tokens
        for user_name, token_amount in self.borrow_token.items():
            for i in range(len(self.borrow_token[user_name])):
                self.borrow_token[user_name][i] *= self._get_daily_interest(self.borrow_interest_rate)

    def update_collateral_factor(self, direction: int):
        new_col_fac = self.collateral_factor + direction * self.col_factor_change_rate
        if not 0 < new_col_fac < 1:
            # TODO: Punish actor for this action by giving him a penalty!
            # And do not update the collateral_factor
            pass
        self.collateral_factor = new_col_fac

    def do_deposit(self, agent_id: int, agent: UserAgent, amount: float) -> None:
        """
        TODO: If agent overspends -> negative reward/punishment
        :param agent_id: Agent's ID
        :param agent: Agent who wants to deposit funds
        :param amount: Amount the agent wants to deposit
        """
        # 1) Check whether agent has enough funds for the deposit
        agent_balance = agent.get_balance(self.token_name)
        if agent_balance < amount:
            logging.info("Agent {} tried to deposit {} into {}, but didn't have enough funds ({})".format(agent_id, amount, self.token_name, agent_balance))
            # TODO: Reward -> negative reward
            return
        # 2) Deduct funds from agents balance
        agent.sub_balance(token_name=self.token_name, amount=amount)
        # 3) Add the funds to the pool
        if self.supply_token.get(str(agent_id)) is None:
            self.supply_token[str(agent_id)] = list()
        self.supply_token[str(agent_id)].append(amount)
        logging.info("Agent {} has deposited {} into {}".format(agent_id, amount, self.token_name))

    def do_withdraw(self, agent_id: int, agent: UserAgent, amount: float) -> None:
        """
        Function removes the oldest
        TODO: If agent withdraws too much funds -> negative reward/punishment
        :param agent_id: Agent's ID
        :param agent: Agent who wants to withdraw funds
        :param amount: Amount the agent wants to withdraw
        """
        # 1) Check whether the user has deposited anything into the pool
        if self.supply_token.get(str(agent_id)) is None or len(self.supply_token.get(str(agent_id))) == 0:
            logging.info("Agent {} tried to withdraw {} from {}, but didn't deposit enough or didn't deposit at all.".format(agent_id, amount, self.token_name))
            # TODO: Reward -> negative reward
            return
        # 2) Remove the funds from the pool
        withdrawal_amount = self.supply_token[str(agent_id)].pop()
        # 3) Add the funds to the agents balance
        agent.add_balance(token_name=self.token_name, amount=withdrawal_amount)
        logging.info("Agent {} has withdrawn {} from {}}".format(agent_id, withdrawal_amount, self.token_name))

    def __repr__(self):
        return (
            "PLFPool(" +
            f"'{self.token_name}', " +
            f"Collateral Factor: {self.collateral_factor:.3f}, " +
            f"Total Borrow: {self.total_borrow_token:.4f}, " +
            f"Total Available Funds: {self.total_supply_token:.4f}" +
            ")"
        )
