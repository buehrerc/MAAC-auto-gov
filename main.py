import json
import logging
from utils.make_env import make_env
from utils.make_agent import make_agent


def init_logger(config):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:[%(levelname)s] >> {%(module)s}: %(message)s",
        handlers=[logging.FileHandler("./log/debug.log"), logging.StreamHandler()],
    )


def training(config):
    init_logger(config)

    logging.info("Start Environment Initialization")
    env = make_env(config)
    logging.info("Finished Environment Initialization")

    logging.info("Start Agent Initialization")
    agents = make_agent(config, env.observation_space, env.action_space)
    logging.info("Finished Agent Initialization")

    logging.info(f"Gym Parameters:: observation_space={env.observation_space.shape}, action_space={env.action_space}")
    env.set_agents(agents)
    env.reset()

    actions = [(0, 1), (0, 2), (0, 4), (0, 12), (0, 4), (0, 3)]

    for i, a in enumerate(actions):
        logging.info(f"Start Round {i} ===============================================================================")
        # state, reward, _, _, _ = env.step(action=env.action_space.sample())
        state, reward, _, _, _ = env.step(action=a)
        logging.debug(f"Environment:: environment_state={state}, reward={reward}")

    logging.info("Finished")


if __name__ == '__main__':
    fs = open("./config/config_template.json")
    config_ = json.load(fs)
    training(config_)
