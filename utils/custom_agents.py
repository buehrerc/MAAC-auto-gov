import logging

import torch
from typing import Dict
from utils.agents import AttentionAgent
from abc import abstractmethod

from envs.market_env.constants import (
    ACTION_USER_DEPOSIT, ACTION_USER_WITHDRAW, ACTION_USER_BORROW, ACTION_USER_REPAY, ACTION_USER_LIQUIDATE,
)


class CustomAgent(AttentionAgent):
    def __init__(
        self,
        action_space: int,
        observation_space: int,
        name: str = "CustomAgent",
        hidden_dim: int = 64,
        lr: float = 0.01,
        onehot_dim: int = 0,
    ):
        super().__init__(
            num_in_pol=observation_space,
            num_out_pol=action_space,
            hidden_dim=hidden_dim,
            lr=lr,
            onehot_dim=onehot_dim
        )
        self.name = name

    def get_name(self) -> str:
        return self.name


class UserAgent(CustomAgent):

    def __init__(
        self,
        action_space: int,
        observation_space: int,
        name: str = "UserAgent",
        hidden_dim: int = 64,
        lr: float = 0.01,
        onehot_dim: int = 0,
        **kwargs
    ):
        super().__init__(action_space=action_space, observation_space=observation_space, name=name,
                         hidden_dim=hidden_dim, lr=lr, onehot_dim=onehot_dim)

    def step(self, obs: torch.Tensor, explore=False) -> torch.Tensor:
        return super().step(obs, explore)

    def __repr__(self):
        return f"UserAgent('{self.name}')"


class GovernanceAgent(CustomAgent):
    def __init__(
            self,
            action_space: int,
            observation_space: int,
            name: str = "GovernanceAgent",
            hidden_dim: int = 64,
            lr: float = 0.01,
            onehot_dim: int = 0,
            **kwargs
    ):
        super().__init__(action_space=action_space, observation_space=observation_space, name=name,
                         hidden_dim=hidden_dim, lr=lr, onehot_dim=onehot_dim)

    def reward(self, lending_protocol, action: str) -> float:
        pass

    def __repr__(self):
        return f"GovernanceAgent('{self.name}')"
