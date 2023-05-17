from utils.env_wrappers import SubprocVecEnv


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


class CustomWrapper(SubprocVecEnv):
    def __init__(self, env_fns, spaces=None):
        super().__init__(env_fns)

    def set_agent(self, agent_list):
        for remote in self.remotes:
            remote.send(('set_agent', agent_list))

    def get_spaces(self):
        return self.remotes[0].send(('get_spaces', None))