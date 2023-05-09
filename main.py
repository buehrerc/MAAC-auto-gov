import json
from utils.make_env import make_env
from utils.make_agent import make_agent


def training(config):
    env = make_env(config)
    agents = make_agent(config, env.observation_space, env.action_space)
    env.set_agents(agents)
    env.reset()

    for i in range(10):
        state, _, _, _, _ = env.step(action=env.action_space.sample())
        print(state)

    print('finished')


if __name__ == '__main__':
    fs = open("./config/config_template.json")
    config_ = json.load(fs)
    training(config_)
