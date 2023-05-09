from typing import List

from envs.market_env.market import Market

from envs.market_env.constants import (
    INITIATOR,
    COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_RB_FACTOR,
    PLF_SPREAD,
)


class PLFPool:
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
    ):
        pass

    def do_deposit(self, i: int):
        pass

    def do_withdraw(self, i: int):
        pass