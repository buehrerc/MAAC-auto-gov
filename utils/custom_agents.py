import torch
from typing import Dict
from utils.agents import AttentionAgent
from abc import abstractmethod


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

    @abstractmethod
    def reward(self) -> float:
        """
        Returns the reward of the agent based on the environment
        """
        pass


class UserAgent(CustomAgent):

    def __init__(
        self,
        action_space: int,
        observation_space: int,
        balance: Dict[str, float],
        name: str = "UserAgent",
        hidden_dim: int = 64,
        lr: float = 0.01,
        onehot_dim: int = 0,
        **kwargs
    ):
        self.balance = balance
        observation_space_large = observation_space + len(self.balance)
        super().__init__(action_space=action_space, observation_space=observation_space_large, name=name,
                         hidden_dim=hidden_dim, lr=lr, onehot_dim=onehot_dim)

    def step(self, obs: torch.Tensor, explore=False):
        obs_large = torch.concatenate([obs, torch.Tensor(self.balance.values())])
        super().step(obs_large, explore)

    def get_balance(self, token_name: str) -> float:
        if token_name not in self.balance.keys():
            self.balance[token_name] = 0
        return self.balance[token_name]

    def add_balance(self, token_name: str, amount: float) -> None:
        if token_name not in self.balance.keys():
            self.balance[token_name] = 0
        self.balance[token_name] += amount

    def sub_balance(self, token_name: str, amount: float) -> None:
        self.balance[token_name] -= amount

    def reward(self) -> float:
        pass

    def __repr__(self):
        return f"UserAgent('{self.name}', balance: {self.balance})"


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

    def reward(self):
        pass

    def __repr__(self):
        return f"GovernanceAgent('{self.name}')"
