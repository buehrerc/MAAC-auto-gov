import logging
import torch
import numpy as np
from gym.core import ObsType
from typing import Dict
from envs.market_env.market import Market
from envs.market_env.constants import (
    PLF_INITIATOR,
    PLF_COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_RB_FACTOR,
    PLF_SPREAD,
    PLF_OBSERVATION_SPACE,
)


class PLFPool:
    def __init__(
        self,
        market: Market,
        token_name: str = "dai",
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        col_factor_change_rate: float = PLF_COLLATERAL_FACTOR_CHANGE_RATE,
        rb_factor: float = PLF_RB_FACTOR,
        spread: float = PLF_SPREAD,
        seed: int = 0
    ):
        assert 0 < collateral_factor < 1, "Collateral Factor must be between 0 and 1"
        # Gym Atributes
        self.observation_space = PLF_OBSERVATION_SPACE

        # General Properties
        self.token_name = token_name
        self.token = market.get_token(self.token_name)

        # Pool Parameters
        self.collateral_factor: float = collateral_factor
        self.col_factor_change_rate: float = col_factor_change_rate
        self.supply_token: Dict[str, float] = {PLF_INITIATOR: initial_starting_funds}
        self.borrow_token: Dict[str, float] = dict()

        # Reward Parameters
        self.previous_reserve: float = 0.0

        self.rb_factor = rb_factor
        self.spread = spread

        # Randomness
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def total_supply_token(self) -> float:
        return sum(self.supply_token.values())

    @property
    def total_borrow_token(self) -> float:
        return sum(self.borrow_token.values())

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
        """
        Function returns the following state:
        - Total amount of supply tokens
        - Total amount of borrow tokens
        - Reserve of the pool
        - Utillization ratio of the pool
        - Collateral factor of the pool
        - Supply interest rate of the pool
        - Borrow interest rate of the pool
        :return: torch.Tensor of the states
        """
        return torch.Tensor([
            self.total_supply_token,    # Supply Token
            self.total_borrow_token,    # Borrow Token
            self.reserve,               # Net Position
            self.utilization_ratio,     # Utilization Ratio
            self.collateral_factor,     # Collateral Factor
            self.supply_interest_rate,  # Supply Interest Rate
            self.borrow_interest_rate   # Borrow Interest Rate
        ])

# =====================================================================================================================
#   POOL GETTER
# =====================================================================================================================
    def get_token_price(self) -> float:
        return self.token.get_price()

    def get_token_name(self) -> str:
        return self.token_name

    def get_collateral_factor(self) -> float:
        return self.collateral_factor

# =====================================================================================================================
#   UPDATE POOL
# =====================================================================================================================
    def update(self) -> ObsType:
        """
        Function is being used to update the pool internal parameters
        :return: state
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
        for supply_key in self.supply_token.keys():
            self.supply_token[supply_key] *= self._get_daily_interest(self.supply_interest_rate)
        # Accrue borrow tokens
        for borrow_key in self.borrow_token.keys():
            self.borrow_token[borrow_key] *= self._get_daily_interest(self.borrow_interest_rate)

# =====================================================================================================================
#   POOL ACTIONS
# =====================================================================================================================
    def update_collateral_factor(self, direction: int) -> (float, bool):
        """
        Update the collateral factor of the pool by increasing or decreasing by a constante rate
        :param direction: -1=decrease, +1=incrase
        :return: reward, success
        """
        new_col_fac = self.collateral_factor + direction * self.col_factor_change_rate
        if not 0 < new_col_fac < 1:
            # TODO: Punish actor for this action by giving him a penalty!
            # And do not update the collateral_factor
            return 0.0, False
        self.collateral_factor = new_col_fac
        return 0.0, True

    def add_supply(self, key: str, amount: float) -> None:
        logging.debug(f"Supply of {amount} was added to pool '{self.token_name}'")
        self.supply_token[key] = amount

    def remove_supply(self, key: str) -> float:
        amount = self.supply_token.pop(key)
        logging.debug(f"Supply of {amount} was removed from pool '{self.token_name}'")
        return amount

    def get_supply(self, key: str) -> float:
        return self.supply_token.get(key)

    def start_borrow(self, key: str, amount: float) -> None:
        logging.debug(f"Borrow of {amount} was taken from pool '{self.token_name}'")
        self.borrow_token[key] = amount

    def return_borrow(self, key: str) -> float:
        amount = self.borrow_token.pop(key)
        logging.debug(f"Borrow of {amount} was returned to pool '{self.token_name}'")
        return amount

    def get_borrow(self, key: str) -> float:
        return self.borrow_token.get(key)

    def __repr__(self):
        return (
            "PLFPool(" +
            f"'{self.token_name}', " +
            f"Collateral Factor: {self.collateral_factor:.3f}, " +
            f"Total Borrow: {self.total_borrow_token:.4f}, " +
            f"Total Available Funds: {self.total_supply_token:.4f}" +
            ")"
        )
