import json
from utils.make_env import make_env


def training(config):
    env = make_env(config)
    env.reset()

    for i in range(10):
        state, _, _, _, _ = env.step(action=env.action_space.sample())
        print(state)


if __name__ == '__main__':
    fs = open("./config/config_template.json")
    config_ = json.load(fs)
    training(config_)
