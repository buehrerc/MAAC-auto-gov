import gym
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

import torch
from gym.core import RenderFrame, ActType, ObsType
from gym import spaces
from envs.market_env.utils import combine_observation_space
from envs.market_env.constants import (
    CONFIG_MARKET,
    CONFIG_TOKEN,
    TOKEN_OBSERVATION_SPACE
)


class Market:
    """
    Market dictates the price of all assets
    The prices adhere a geometric Brownian motion with zero drift and a custom volatility
    """
    def __init__(
        self,
        config: Dict,
        seed: int,
    ):
        self.market_config = config[CONFIG_MARKET]
        self.tokens: Dict[str, Token] = dict()
        self.observation_space: spaces.Space = spaces.Space()
        self.seed = seed
        self.reset()

    def reset(self) -> ObsType:
        """
        Resets the market to the initial parameters
        Token object does not have a reset function, since the Tokens are just reinitialized
        :return:
        """
        # Make sure that a fresh seed is used for each episode
        self.seed += len(self.market_config[CONFIG_TOKEN])
        token_list = [self._initialize_token(param, self.seed+i)
                      for i, param in enumerate(self.market_config[CONFIG_TOKEN])]

        token_names = [t.get_name() for t in token_list]
        self.tokens = dict(zip(token_names, token_list))
        self.observation_space = combine_observation_space(token_list)

        observation_spaces = [t.observation_space for t in token_list]
        self.observation_space = spaces.Box(
            low=np.concatenate([o.low for o in observation_spaces]),
            high=np.concatenate([o.high for o in observation_spaces])
        )
        return self._get_state()

    def _get_state(self) -> ObsType:
        return torch.cat([token.get_state() for token in self.tokens.values()])

    @staticmethod
    def _initialize_token(param, seed):
        return Token(seed=seed, **param)

    def update(self) -> ObsType:
        """
        Updates all prices of the asset token within the market
        :return: None
        """
        return torch.cat([token.update() for token in self.tokens.values()])

    def get_prices(self) -> Dict[str, float]:
        return dict([(name, token.get_price()) for name, token in self.tokens.items()])

    def get_token(self, name: str = None):
        if name is None:
            return self.tokens
        try:
            return self.tokens[name]
        except KeyError:
            raise KeyError("Token {} does not exist, thus no PLFPool could be initialized".format(name))

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def __repr__(self) -> str:
        return "Market(" + repr(self.tokens) + ")"


class Token:
    def __init__(
        self,
        name: str = "dai",
        price: float = 1000.0,
        borrow_interest_rate: float = 0.05,
        supply_interest_rate: float = 0.15,
        asset_volatility: float = 0.1,
        seed: int = 0
    ) -> None:
        """

        :param name: Name of the token
        :param price: Price of the token
        :param borrow_interest_rate: Competing interest rate for borrowing this asset (can be thought of as the interest rate offered by another PLF)
        :param supply_interest_rate: Competing interest rate for supply this asset (can be thought of as the interest rate offered by another PLF)
        :param asset_volatility: Volatility of this token/asset
        :param seed: Random seed
        """
        # Gym Attributes
        self.observation_space = TOKEN_OBSERVATION_SPACE

        # General Attributes
        self.name = name
        self.price = price
        self.borrow_interest_rate = borrow_interest_rate
        self.supply_interest_rate = supply_interest_rate
        self.asset_volatility = asset_volatility
        self.rng = np.random.RandomState(seed)

    def get_name(self) -> str:
        return self.name

    def get_price(self) -> float:
        return self.price

    def get_state(self) -> torch.Tensor:
        return torch.Tensor([
            self.price,
            self.borrow_interest_rate,
            self.supply_interest_rate,
            self.asset_volatility
        ])

    def get_borrow_interest_rate(self):
        return self.borrow_interest_rate

    def get_supply_interest_rate(self):
        return self.supply_interest_rate

    def update(self) -> torch.Tensor:
        # Asset price adheres geometric Brownian motion with zero drift
        new_price = self.price * np.exp(self.asset_volatility * self.rng.normal(0, 1))
        assert new_price > 0, "asset price cannot be negative."
        self.price = new_price
        return self.get_state()

    def __repr__(self):
        return f"Token(Name: '{self.name}', Price: {self.price:.2f}, Sigma: {self.asset_volatility:.2f})"
