from envs.market_env.lending_protocol import LendingProtocol
from envs.market_env.constants import (
    CONFIG_AGENT_TYPE_GOVERNANCE,
    CONFIG_AGENT_TYPE_USER,
    REWARD_ILLEGAL_ACTION
)


def reward_function(agent_type: str, lending_protocol: LendingProtocol, illegal_action: bool) -> float:
    if agent_type == CONFIG_AGENT_TYPE_GOVERNANCE:
        return governance_reward(lending_protocol, illegal_action)
    elif agent_type == CONFIG_AGENT_TYPE_USER:
        return user_reward(lending_protocol, illegal_action)
    else:
        raise NotImplementedError("Agent type {} is unknown".format(agent_type))


def governance_reward(lending_protocol: LendingProtocol, illegal_action: bool) -> float:
    """
    The standard reward for the governance agent is its lending protocol's revenue
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION

    return sum([plf_pool.get_revenue() for plf_pool in lending_protocol.plf_pools])


def user_reward(lending_protocol: LendingProtocol, illegal_action: bool) -> float:
    """
    The standard reward for the user agent is its maximum exposure to the individual lending pools
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION
    # TODO: Implement maximum exposure
    return 1.0
