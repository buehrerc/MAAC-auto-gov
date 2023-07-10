import os
import json
import random
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torch.autograd import Variable
from gym.spaces import Box
from pathlib import Path
from algorithms.custom_attention_sac import CustomAttentionSAC
from utils.make_agent import make_agent
from utils.custom_wrappers import CustomWrapper, CustomDummyWrapper
from algorithms.attention_sac import AttentionSAC
from envs.market_env.env import MultiAgentEnv
from envs.market_env.constants import CONFIG_PARAM
from envs.market_env.utils import generate_state_mapping
from utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter


def init_logger(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:[%(levelname)s] >> {%(module)s}: %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "debug.log")), logging.StreamHandler()],
    )


def init_params(config, env_config):
    # Initialize directory for logs and model
    model_dir = Path('./models') / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    # Logger
    logger = SummaryWriter(str(log_dir))
    state_mapping = generate_state_mapping(env_config)

    # Fix randomness
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Fix Exploration Limit based on configs
    exploration_limit = config.buffer_length // config.n_rollout_threads

    return logger, run_num, run_dir, log_dir, state_mapping, exploration_limit


def init_env(env_config, seed):
    logging.info("Start Environment Initialization")
    env = MultiAgentEnv(env_config, seed)
    logging.info("Finished Environment Initialization")
    logging.info(f"Gym Parameters:: observation_space={env.observation_space.shape}, action_space={env.action_space}")
    return env


def make_parallel_env(env_config, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env_():
            env = MultiAgentEnv(env_config, seed+rank)
            return env
        return init_env_
    if n_rollout_threads == 1:
        return CustomDummyWrapper(get_env_fn(0))
    else:
        return CustomWrapper([get_env_fn(i) for i in range(n_rollout_threads)])


def train(
    env: CustomWrapper,
    model: AttentionSAC,
    replay_buffer: ReplayBuffer,
    state_mapping: List[str],
    logger: SummaryWriter,
    config,
    run_dir: str,
):
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        logging.info("Episodes %i-%i of %i" % (ep_i + 1,
                                               ep_i + 1 + config.n_rollout_threads,
                                               config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # CBUE MODIFICATION: replaced obs[:, i] with obs, since all agents have the same observation space
            torch_obs = [Variable(torch.Tensor(np.vstack(obs)),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # CBUE MODIFICATION: different mapping of the actions.
            # rearrange actions to be per environment
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            actions = [[np.where(ac[i] == 1)[0][0] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # CBUE MODIFICATION: if assertions fail, the environment becomes unusable -> reset the environment
            try:
                next_obs, rewards, dones, infos = env.step(actions)
            except AssertionError as ae:
                logging.info(str(ae))
                _ = env.reset()
                break
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        # Recap Episode
        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        logging.info(f"Average Reward: {ep_rews}")

        for name, value in zip(state_mapping, replay_buffer.get_buffer_data(config.episode_length * config.n_rollout_threads).mean(0)):
            logger.add_scalar(name, value, ep_i)

        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads and ep_i > 1:
            # Logging of states
            os.makedirs(run_dir / 'plots', exist_ok=True)
            for name, value in zip(state_mapping,
                                   replay_buffer.get_buffer_data(config.episode_length * config.n_rollout_threads)[
                                   ::config.n_rollout_threads].T):
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                ax.plot(np.arange(len(value)), value)
                ax.set_title(f'episode_{ep_i}' + "_" + name.replace("/", "_"))
                logger.add_figure('matplotlib/' + name, fig, ep_i)

            # Logging of weights
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')


def run(config, env_config):
    logger, run_num, run_dir, log_dir, state_mapping, exploration_limit = init_params(config, env_config)
    init_logger(log_dir)

    env = make_parallel_env(env_config, config.n_rollout_threads, config.seed)
    obsp, acsp = env.get_spaces()
    agents = make_agent(env_config, obsp, acsp, exploration_limit)

    model = CustomAttentionSAC(
        env=env,
        agents=agents,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale
    )
    replay_buffer = ReplayBuffer(
        max_steps=config.buffer_length,
        num_agents=model.nagents,
        obs_dims=[env.observation_space.shape[0]] * len(env.action_space),
        ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    train(env, model, replay_buffer, state_mapping, logger, config, run_dir)

    model.save(run_dir / 'model.pt')
    env_config['args'] = vars(config)
    with open(log_dir / "config.json", "w") as fp:
        json.dump(env_config, fp)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


def get_env_config(config):
    fs = open(config.config)
    env_config = json.load(fs)
    fs.close()
    # Overwrite config with json params
    for key, value in env_config[CONFIG_PARAM].items():
        setattr(config, key, value)
    return config, env_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of directory to store model/training contents")
    parser.add_argument("--config", default="./config/default_config.json")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.01, type=float)
    parser.add_argument("--q_lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--seed", default=42, type=int)

    config_ = parser.parse_args()
    config_, env_config_ = get_env_config(config_)
    run(config_, env_config_)
