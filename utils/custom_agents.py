import torch
from utils.agents import AttentionAgent


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

    def __repr__(self):
        return f"GovernanceAgent('{self.name}')"
