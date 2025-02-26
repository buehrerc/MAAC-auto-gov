from typing import List

from algorithms.attention_sac import AttentionSAC
from utils.agents import AttentionAgent


class CustomAttentionSAC(AttentionSAC):
    """
    Extension of AttentionSAC by providing a custom initialization method for the object
    """
    def __init__(
        self,
        env,
        agents: List[AttentionAgent],
        gamma: float = 0.95,
        tau: float = 0.01,
        pi_lr: float = 0.01,
        q_lr: float = 0.01,
        reward_scale: int = 10,
        critic_hidden_dim: int = 128,
        attend_heads: int = 4,
        **kwargs
    ):
        """
        Instantiate instance of this class from multiagent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        sa_size = list()
        agent_init_params = list()
        for acsp in env.action_space:
            # We assume that all agents have the same observation_space
            sa_size.append((env.observation_space.shape[0], acsp.n))
            agent_init_params.append({'num_in_pol': env.observation_space.shape[0],
                                      'num_out_pol': acsp.n})

        super().__init__(agent_init_params=[], sa_size=sa_size,
                         gamma=gamma, tau=tau, pi_lr=pi_lr, q_lr=q_lr,
                         reward_scale=reward_scale,
                         critic_hidden_dim=critic_hidden_dim, attend_heads=attend_heads,
                         **kwargs)
        # Overwrite the agents
        self.agents = agents
        self.init_dict = {'gamma': gamma, 'tau': tau,
                          'pi_lr': pi_lr, 'q_lr': q_lr,
                          'reward_scale': reward_scale,
                          'critic_hidden_dim': critic_hidden_dim,
                          'attend_heads': attend_heads,
                          'agent_init_params': agent_init_params,
                          'sa_size': sa_size}

    def step(self, observations, explore=False, exploration_rate=0.1):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore, exploration_rate=exploration_rate)
                for a, obs in zip(self.agents, observations)]
