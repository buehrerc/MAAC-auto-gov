from typing import Tuple, Dict

import torch
import numpy as np
from gym.core import ActType, ObsType

from utils.env_wrappers import SubprocVecEnv
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            remote.send(env.agent_mask)
        else:
            raise NotImplementedError()


class CustomWrapper(SubprocVecEnv):
    def __init__(self, env_fns):
        """
        :param env_fns: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def get_spaces(self):
        self.remotes[0].send(('get_spaces', None))
        return self.remotes[0].recv()


class CustomDummyWrapper(VecEnv):
    """
    Wrapper reshapes the in- and outputs to comply with the general learning pipeline.
    """
    def __init__(self, env_fn):
        self.env = env_fn()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.actions = None
        super().__init__(1, self.observation_space, self.action_space)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self) -> Tuple[ObsType, torch.Tensor, np.array, Dict]:
        # Unwrap action
        action_unwrapped = self.actions[0]
        obs, reward, done, log = self.env.step(action_unwrapped)
        # Wrap all results
        return obs.unsqueeze(0), reward.unsqueeze(0), np.array([done]), log

    def get_spaces(self) -> Tuple:
        return self.observation_space, self.action_space

    def reset(self) -> ObsType:
        return self.env.reset().unsqueeze(0)

    def close(self):
        pass

