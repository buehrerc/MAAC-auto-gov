import logging
import torch
from gym.core import ObsType
from typing import Dict, List
from envs.market_env.market import Market
from envs.market_env.constants import (
    PLF_INITIATOR,
    PLF_COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_RB_FACTOR,
    PLF_SPREAD,
    PLF_OBSERVATION_SPACE,
    PLF_INTEREST_CHANGE_RATE,
    PLF_FEE,
    PLF_OPTIMAL_UTILIZATION_RATIO, PLF_STABLE_BORROW_SLOPE_1, PLF_STABLE_BORROW_SLOPE_2,
    PLF_BASE_BORROW_RATE, PLF_VARIABLE_BORROW_SLOPE_1, PLF_VARIABLE_BORROW_SLOPE_2,
    LP_BORROW_SAFETY_MARGIN,
)


class PLFPool:

    def __init__(
            self,
            market: Market,
            agent_balance: List[Dict],
            owner: int,
            token_name: str = "dai",
            initial_starting_funds: float = 1000,
            collateral_factor: float = 0.85,
            col_factor_change_rate: float = PLF_COLLATERAL_FACTOR_CHANGE_RATE,
            interest_change_rate: float = PLF_INTEREST_CHANGE_RATE,
            rb_factor: float = PLF_RB_FACTOR,
            spread: float = PLF_SPREAD,
    ):
        assert 0 < collateral_factor < 1, "Collateral Factor must be between 0 and 1"
        # Gym Atributes
        self.observation_space = PLF_OBSERVATION_SPACE

        # General Properties
        self.token_name = token_name
        self.token = market.get_token(self.token_name)
        self.agent_balance = agent_balance
        self.owner = owner

        # Pool Parameters
        self.collateral_factor: float = collateral_factor
        self.col_factor_change_rate: float = col_factor_change_rate
        self.supply_token: Dict[str, float] = {PLF_INITIATOR: initial_starting_funds}
        # Set the initial utilization ratio to the optimal utilization ratio
        self.borrow_token: Dict[str, float] = {PLF_INITIATOR: initial_starting_funds * PLF_OPTIMAL_UTILIZATION_RATIO}

        # Interest Rate Model
        self.interest_change_rate = interest_change_rate
        self.optimal_utilization_ratio: float = PLF_OPTIMAL_UTILIZATION_RATIO
        self.base_borrow_rate: float = PLF_BASE_BORROW_RATE
        self.stable_borrow_slope_1: float = PLF_STABLE_BORROW_SLOPE_1
        self.stable_borrow_slope_2: float = PLF_STABLE_BORROW_SLOPE_2
        self.variable_borrow_slope_1: float = PLF_VARIABLE_BORROW_SLOPE_1
        self.variable_borrow_slope_2: float = PLF_VARIABLE_BORROW_SLOPE_2

        # Reward Parameters
        self.previous_reserve_value: float = 0.0

        self.rb_factor = rb_factor
        self.spread = spread
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
        """
        AAVE interest rate model
        """
        assert self.utilization_ratio > -1e-9, "Utilization Ratio must be non-negative"
        constrained_util_ratio = max(0, min(self.utilization_ratio, 0.97))
        return self.borrow_interest_rate * constrained_util_ratio * (1 - self.spread)

    @property
    def borrow_interest_rate(self) -> float:
        """
        AAVE interest rate model
        For predictability, the stable interest rate model is being used
        """
        assert self.utilization_ratio > -1e-9, "Utilization Ratio must be non-negative"
        if self.utilization_ratio > self.optimal_utilization_ratio:
            excess_utilization_ratio = (self.utilization_ratio - self.optimal_utilization_ratio) / (
                    1 - self.optimal_utilization_ratio)
            stable_borrow_interest_rate = (
                    self.base_borrow_rate
                    + self.stable_borrow_slope_1
                    + self.stable_borrow_slope_2 * excess_utilization_ratio
            )
            # variable_borrow_interest_rate = (
            #         self.base_borrow_rate
            #         + self.variable_borrow_slope_1
            #         + self.variable_borrow_slope_2 * excess_utilization_ratio
            # )
        else:
            actual_to_optimal = self.utilization_ratio / self.optimal_utilization_ratio
            stable_borrow_interest_rate = (
                    self.base_borrow_rate
                    + self.stable_borrow_slope_1 * actual_to_optimal
            )
            # variable_borrow_interest_rate = (
            #         self.base_borrow_rate
            #         + actual_to_optimal * self.variable_borrow_slope_1
            # )
        return stable_borrow_interest_rate

    @property
    def reserve(self) -> float:
        return self.total_supply_token - self.total_borrow_token

    @staticmethod
    def _get_daily_interest(interest: float) -> float:
        return (1 + interest) ** (1 / 365)

    def get_revenue(self) -> float:
        revenue = self.reserve * self.get_token_price() - self.previous_reserve_value
        self.previous_reserve_value = self.reserve * self.get_token_price()
        return revenue

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
            self.borrow_interest_rate,  # Borrow Interest Rate
            self.base_borrow_rate,      # Interest Model Base Rate
            self.optimal_utilization_ratio, # Interest Model Optimal Utilization Ratio
            self.stable_borrow_slope_1, # Interest Model Slope 1
            self.stable_borrow_slope_2, # Interest Model Slope 2
        ])

    # =====================================================================================================================
    #   POOL GETTER & SETTER
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
        return self.get_state()

    def accrue_interest(self) -> None:
        """
        Accrue interest to all users in the pool
        :return: None
        """
        # Accrue interest tokens
        for supply_hash in self.supply_token.keys():
            if supply_hash == PLF_INITIATOR:  # No interest is accrued for initialization supply
                continue
            daily_interest = self.supply_token[supply_hash] * (self._get_daily_interest(self.supply_interest_rate) - 1)
            daily_interest_fee = daily_interest * PLF_FEE
            assert daily_interest_fee >= 0, "Fee has to be positive"
            self.agent_balance[self.owner][self.token_name] += daily_interest_fee
            self.supply_token[supply_hash] += daily_interest - daily_interest_fee

        # Accrue borrow tokens
        for borrow_hash in self.borrow_token.keys():
            if borrow_hash == PLF_INITIATOR:  # No interest is accrued for initialization borrow
                continue
            daily_interest = self.borrow_token[borrow_hash] * (self._get_daily_interest(self.borrow_interest_rate) - 1)
            daily_interest_fee = daily_interest * PLF_FEE
            assert daily_interest_fee >= 0, "Fee has to be positive"
            self.agent_balance[self.owner][self.token_name] += daily_interest_fee
            self.borrow_token[borrow_hash] += daily_interest - daily_interest_fee

    # =====================================================================================================================
    #   POOL ACTIONS
    # =====================================================================================================================
    def update_collateral_factor(self, direction: int) -> bool:
        """
        Update the collateral factor of the pool by increasing or decreasing by a constante rate
        :param direction: -1=decrease, +1=increase

        :return: True: illegal_action, False: legal_action
        """
        new_col_fac = self.collateral_factor + direction * self.col_factor_change_rate
        if not LP_BORROW_SAFETY_MARGIN < new_col_fac < 1:
            # Lower limit prevents negative loan amounts
            return True
        self.collateral_factor = new_col_fac
        return False

    def update_interest_model(self, stable_borrow_slope_1: int = 0, stable_borrow_slope_2: int = 0) -> bool:
        """
        Update the parameters of the interest model.
        :param stable_borrow_slope_1: -1=decrease, +1=increase
        :param stable_borrow_slope_2: -1=decrease, +1=increase
        """
        for attr_name, direction in [("stable_borrow_slope_1", stable_borrow_slope_1), ("stable_borrow_slope_2", stable_borrow_slope_2)]:
            new_slope = getattr(self, attr_name) + self.interest_change_rate * direction
            if not 0 < new_slope < 1:
                # And do not update the collateral_factor
                return True
            setattr(self, attr_name, new_slope)
        return False

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
