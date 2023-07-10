import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from gym.spaces import Box
from matplotlib import pyplot as plt
from torch.autograd import Variable
from algorithms.attention_sac import AttentionSAC
from envs.market_env.utils import generate_state_mapping
from main import make_parallel_env
from envs.market_env.constants import CONFIG_PARAM
from utils.buffer import ReplayBuffer
from tensorboardX import SummaryWriter


NUM_ANALYSIS_RUNS = 10


def init(config):
    base_dir = Path("../models/" + config.model)
    log_dir = Path("../results/" + config.model)
    os.makedirs(log_dir, exist_ok=True)

    model_path = base_dir / "model.pt"
    fs = open(base_dir / "logs" / "config.json")
    env_config = json.load(fs)
    fs.close()
    # Overwrite config with json params
    for key, value in env_config[CONFIG_PARAM].items():
        setattr(config, key, value)

    env = make_parallel_env(env_config, n_rollout_threads=1, seed=0)
    model = AttentionSAC.init_from_save(model_path, load_critic=False)
    n_episodes = NUM_ANALYSIS_RUNS//10 if config.store_image else NUM_ANALYSIS_RUNS
    replay_buffer = ReplayBuffer(
        max_steps=config.episode_length * n_episodes,
        num_agents=model.nagents,
        obs_dims=[env.observation_space.shape[0]] * len(env.action_space),
        ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                 for acsp in env.action_space]
    )
    logger = SummaryWriter(str(log_dir))
    state_mapping = generate_state_mapping(env_config)

    return env, model, replay_buffer, logger, state_mapping, log_dir


def main(config):
    env, model, replay_buffer, logger, state_mapping, log_dir = init(config)
    model.prep_rollouts(device='cpu')
    n_episodes = NUM_ANALYSIS_RUNS//10 if config.store_image else NUM_ANALYSIS_RUNS

    for run_i in range(n_episodes):
        print(f"Episode {run_i+1} of {n_episodes}")
        obs = env.reset()
        for i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs)),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            torch_agent_actions = model.step(torch_obs, explore=False)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[np.where(ac[i] == 1)[0][0] for ac in agent_actions] for i in range(1)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs

        if config.store_image:
            for name, value in zip(state_mapping,
                                   replay_buffer.get_buffer_data(config.episode_length).T):
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                ax.plot(np.arange(len(value)), value)
                ax.set_title(name.replace("/", "_"))
                logger.add_figure('matplotlib/' + name, fig, run_i)

            for name, value in zip(['reward/agent_0', 'reward/agent_1'], replay_buffer.rew_buffs):
                inds = np.arange(replay_buffer.curr_i - config.episode_length, replay_buffer.curr_i)
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                ax.plot(np.arange(len(inds)), value[inds])
                ax.set_title(name.replace("/", "_"))
                logger.add_figure('matplotlib/' + name, fig, run_i)

            for name, value in zip(['action/agent_0', 'action/agent_1'], replay_buffer.ac_buffs):
                inds = np.arange(replay_buffer.curr_i - config.episode_length, replay_buffer.curr_i)
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                ax.plot(np.arange(len(inds)), np.argmax(value[inds], axis=1))
                ax.set_title(name.replace("/", "_"))
                logger.add_figure('matplotlib/' + name, fig, run_i)

    # Save first episode as a DataFrame
    data_list = list()
    for i, ep_data in enumerate(replay_buffer.obs_buffs):
        tmp = pd.DataFrame(
            data=ep_data[:config.episode_length],
            columns=[f'agent{i}/'+s for s in state_mapping]
        )
        tmp[f'reward/agent_{i}'] = replay_buffer.rew_buffs[i][:config.episode_length]
        tmp[f'action/agent_{i}'] = np.argmax(replay_buffer.ac_buffs[i][:config.episode_length], axis=1)
        data_list.append(tmp)
    df = pd.concat(data_list, axis=1).fillna(0)
    df.to_csv(log_dir / 'episode_data.csv')
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Name of directory with checkpoint files")
    parser.add_argument("--store_image", action="store_true")
    config_ = parser.parse_args()

    main(config_)
