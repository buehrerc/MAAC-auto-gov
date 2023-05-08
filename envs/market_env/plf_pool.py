import numpy as np
import torch
from typing import List, Tuple
from gym.core import ObsType
from gym import spaces
from envs.market_env.market import Market

from envs.market_env.constants import (
    INITIATOR,
    COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_RB_FACTOR,
    PLF_SPREAD,
    PLF_OBSERVATION_SPACE,
    PLF_ACTION_SPACE,
    PLF_GOVERNANCE_ACTION_MAPPING,
    PLF_USER_ACTION_MAPPING,
    PLF_STEP_SIZE,
    CONFIG_AGENT_TYPE_USER,
    CONFIG_AGENT_TYPE_GOVERNANCE,
)


class PLFPool:
    """
    Class implements a lending pool.
    States of the PLFPool include:
    + Total Available Funds
    + Supply Token
    + Borrow Token
    + Net Position
    + Utilization Ratio
    + Collateral Factor
    + Supply Interest Rate
    + Borrow Interest Rate
    """
    def __init__(
        self,
        market: Market,
        agent_mask: List[str],
        token_name: str = "dai",
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        col_factor_change_rate: float = COLLATERAL_FACTOR_CHANGE_RATE,
        rb_factor: float = PLF_RB_FACTOR,
        spread: float = PLF_SPREAD,
        seed: int = 0
    ) -> None:
        """
        :param market: Market which holds the tokens which are traded in this specific pool
        :param initial_starting_funds: The initial amount of funds that is put into the liquidity pool
        :param collateral_factor: The collateral factor of the liquidity pool
        :param seed: Randomness Seed
        """
        assert 0 < collateral_factor < 1, "Collateral Factor must be between 0 and 1"

        # Gym Attributes
        self.observation_space = PLF_OBSERVATION_SPACE
        self.action_space = self._get_action_space(agent_mask)

        # General Properties
        self.token_name = token_name
        self.agent_mask = agent_mask
        self.token = market.get_token(self.token_name)
        self.col_factor_change_rate = col_factor_change_rate
        self.asset_price = self.token.get_price()
        self.asset_price_history: List[float] = [self.token.get_price()]
        self.collateral_factor: float = collateral_factor

        # Interest Tokens
        self.user_interest_token: dict[str, float] = {INITIATOR: initial_starting_funds}
        self.user_borrow_token: dict[str, float] = {INITIATOR: 0.0}

        # KPI Properties
        self.profit: float = 0.0
        self.total_borrowed_funds: float = 0.0
        self.total_available_funds: float = initial_starting_funds
        self.previous_reserve: float = 0.0

        # PLF Pool Constants
        self.rb_factor = rb_factor
        self.spread = spread

        # Randomness
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @staticmethod
    def _get_action_space(agent_mask):
        return spaces.Tuple([PLF_ACTION_SPACE[agent_type] for agent_type in agent_mask])

    @property
    def utilization_ratio(self) -> float:
        if self.total_interest_tokens == 0:
            return 0
        return self.total_borrow_tokens / self.total_interest_tokens

    @property
    def total_interest_tokens(self) -> float:
        return sum(self.user_interest_token.values())

    @property
    def total_borrow_tokens(self) -> float:
        return sum(self.user_borrow_token.values())

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
        return self.total_borrow_tokens + self.total_available_funds - self.total_interest_tokens

    @property
    def reward(self) -> float:
        return self.reserve - self.previous_reserve

    @staticmethod
    def _get_daily_interest(amount: float) -> float:
        return (1 + amount) ** (1 / 365)

    def get_profit(self) -> float:
        return self.reward

    def get_token_name(self):
        return self.token_name

    def get_state(self) -> torch.Tensor:
        return torch.Tensor([
            self.total_available_funds,  # Total Available Funds
            self.total_interest_tokens,  # Supply Token
            self.total_borrow_tokens,    # Borrow Token
            self.reserve,                # Net Position
            self.utilization_ratio,      # Utilization Ratio
            self.collateral_factor,      # Collateral Factor
            self.supply_interest_rate,   # Supply Interest Rate
            self.borrow_interest_rate    # Borrow Interest Rate
        ])

    def step(self, action: torch.Tensor) -> Tuple[ObsType, torch.Tensor, bool, bool, dict]:
        for i, agent_type in enumerate(self.agent_mask):
            if agent_type == CONFIG_AGENT_TYPE_GOVERNANCE:
                self.governance_step(i, int(action[i]))
            elif agent_type == CONFIG_AGENT_TYPE_USER:
                self.agent_step(i, int(action[i]))
            else:
                raise NotImplementedError("Agent Type '{}' is unknown".format(agent_type))

        # Updating the PLFPool properties as well
        self.accrue_interest()
        self.previous_reserve = self.reserve

        return self.step_result()

    def step_result(self):
        return (
            self.get_state(),
            torch.Tensor(),  # TODO: include the reward for the individual agents
            False,
            False,
            dict()
        )

    def agent_step(self, agent_nr: int, action: int):
        """
        The PLF has to react to the following actions:
        > Deposit
        > Withdraw
        > Borrow
        > Repay
        > Liquidate
        :param agent_nr:
        :param action:
        :return:
        """
        getattr(self, "do_" + PLF_USER_ACTION_MAPPING[action])()

    def do_no_action(self):
        pass

    def do_deposit(self):
        """
        Function
        1) deducts money from agents balance
        2) adds it to the pools balance
        3) mints supply interest token

        TODO:
        If agent overspends -> negative reward/punishment
        :return:
        """
        print("Agent deposited")

    def do_withdraw(self):
        """
        1) adds money to agent's balance
        2) removes money from the pool's balance
        3) supply interest tokens are burned

        TODO:
        If agent overspends -> negative reward/punishment
        :return:
        """
        print("Agent withdrew")

    def do_borrow(self):
        """
        1) removes money from the pool's balance
        2) add money to the user's balance
        3) mints borrow interest tokens
        :return:
        """

        print("Agent borrowed")


    def do_repay(self):
        """
        1) adds money to the pool's balance
        2) removes money from the user's balance
        3) burns borrow interest tokens
        :return:
        """
        print("Agent repaid")

    def do_liquidate(self):
        """
        In the simplified training environment of Xu Et al., the
        liquidation of a borrow position updates the pool states
        just like a repayment does.
        :return:
        """
        print("Agent liquidated")
        self.do_repay()

    def governance_step(self, agent_nr: int, action: int) -> None:
        """
        Governance Agent performs a step on
        Action encoding:
        > -1: lower collateral factor
        > 0: keep collateral factor
        > 1: raise collateral factor
        :param action: Encoded action
        :return: None
        """
        action_mapped = PLF_GOVERNANCE_ACTION_MAPPING[action]
        new_col_fac = self.collateral_factor + action_mapped*self.col_factor_change_rate
        if not 0 < new_col_fac < 1:
            # TODO: Punish actor for this action by giving him a penalty!
            # And do not update the collateral_factor
            pass
        self.collateral_factor = new_col_fac

    def accrue_interest(self) -> None:
        """
        Accrue interest to all users in the pool
        :return: None
        """
        # Accrue interest tokens
        for user_name, token_amount in self.user_interest_token.items():
            self.user_interest_token[user_name] *= self._get_daily_interest(self.supply_interest_rate)
        # Accrue borrow tokens
        for user_name, token_amount in self.user_borrow_token.items():
            self.user_borrow_token[user_name] *= self._get_daily_interest(self.borrow_interest_rate)
        # TODO: These tokens belong to the users -> They should be updated in their wallets as well

    def __repr__(self):
        return (
            "PLFPool(" +
            f"'{self.token_name}', " +
            f"Collateral Factor: {self.collateral_factor:.2f}, "+
            f"Total Borrow: {self.total_borrowed_funds:.4f}, " +
            f"Total Available Funds: {self.total_available_funds:.4f}" +
            ")"
        )
