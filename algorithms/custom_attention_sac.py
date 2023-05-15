from algorithms.attention_sac import AttentionSAC
from envs.market_env.env import MultiAgentEnv
from torch.optim import Adam
from utils.misc import hard_update
from utils.critics import AttentionCritic


class CustomAttentionSAC(AttentionSAC):
    """
    Extension of AttentionSAC by providing an additional method of initializing the object
    """
    def init_from_custom(
        self,
        env: MultiAgentEnv,
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
        Instantiate instanfe of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        sa_size = list()
        for acsp, obsp in env.action_space:
            # We assume that all agents have the same observation_space
            sa_size.append((env.observation_space.shape[0], acsp.n))

        self.nagents = len(sa_size)

        self.agents = env.agent_list
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
