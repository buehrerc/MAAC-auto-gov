import torch
from torch.optim import Adam

from utils.agents import AttentionAgent
from utils.custom_policies import CustomDiscretePolicy
from utils.misc import hard_update


class CustomAgent(AttentionAgent):
    def __init__(
        self,
        action_space: int,
        observation_space: int,
        name: str = "CustomAgent",
        hidden_dim: int = 64,
        lr: float = 0.01,
        onehot_dim: int = 0,
        exploration_limit: int = 100,
        **kwargs
    ):
        self.policy = CustomDiscretePolicy(input_dim=observation_space,
                                           output_dim=action_space,
                                           exploration_limit=exploration_limit,
                                           hidden_dim=hidden_dim,
                                           onehot_dim=onehot_dim)
        self.target_policy = CustomDiscretePolicy(input_dim=observation_space,
                                                  output_dim=action_space,
                                                  exploration_limit=exploration_limit,
                                                  hidden_dim=hidden_dim,
                                                  onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.name = name

    def __repr__(self):
        return f"CustomAgent('{self.name}')"
