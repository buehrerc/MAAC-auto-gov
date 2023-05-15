import logging
from typing import Dict, List
from gym import spaces
from envs.market_env.constants import (
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    CONFIG_AGENT_TYPE_USER,
    CONFIG_AGENT_TYPE_GOVERNANCE
)
from utils.agents import AttentionAgent


def make_agent(
    config: Dict,
    observation_space: spaces.Space,
    action_space: spaces.Space
) -> List[AttentionAgent]:
    from utils.custom_agents import UserAgent, GovernanceAgent

    # Convert the spaces into more accessible types
    action_space = list(action_space)
    observation_space = observation_space.shape[0]

    assert len(config[CONFIG_AGENT]) == len(action_space), "Action Space is not properly set up"

    agent_list = list()
    for agent_config, agent_action_space in zip(config[CONFIG_AGENT], action_space):
        if agent_config[CONFIG_AGENT_TYPE] == CONFIG_AGENT_TYPE_GOVERNANCE:
            new_agent = GovernanceAgent(action_space=agent_action_space.n,
                                        observation_space=observation_space,
                                        **agent_config)
        elif agent_config[CONFIG_AGENT_TYPE] == CONFIG_AGENT_TYPE_USER:
            new_agent = UserAgent(action_space=agent_action_space.n,
                                  observation_space=observation_space,
                                  **agent_config)
        else:
            raise KeyError("Agent type {} is unknown".format(agent_config[CONFIG_AGENT_TYPE]))
        logging.info("Initialization:: " + str(new_agent))
        agent_list.append(new_agent)
    return agent_list