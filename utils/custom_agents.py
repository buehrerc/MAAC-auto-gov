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
        **kwargs
    ):
        super().__init__(
            num_in_pol=observation_space,
            num_out_pol=action_space,
            hidden_dim=hidden_dim,
            lr=lr,
            onehot_dim=onehot_dim
        )
        self.name = name

    def __repr__(self):
        return f"CustomAgent('{self.name}')"
