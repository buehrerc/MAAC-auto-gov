from typing import List, Tuple

from envs.market_env.constants import (
    CONFIG_AGENT,
    REWARD_TYPE_PROTOCOL_REVENUE,
    REWARD_TYPE_MAXIMUM_EXPOSURE,
    REWARD_TYPE_PROFIT,
    REWARD_ILLEGAL_ACTION
)


def reward_function(
    agent_id: int,
    reward_type: List[Tuple],
    env,
    illegal_action: bool,
):
    """

    Supported reward_types:
        + protocol_revenue
        + maximum_exposure
        + profit

    :param agent_id: Id of agent whose reward is computed
    :param reward_type: List of reward functions: [[weight, reward_type]+]
    :param env: Environment to compute the reward function on
    :param illegal_action: True: Agent as performed an illegal action,
                           False: Agent didn't perform an illegal action
    :return: reward
    """
    return sum([
        weight * reward_function_by_type(agent_id, rt, env, illegal_action) for weight, rt in reward_type
    ])


def reward_function_by_type(
    agent_id: int,
    reward_type: str,
    env,
    illegal_action: bool
) -> float:
    """
    Supported reward_types:
        + protocol_revenue
        + maximum_exposure

    :param agent_id: Id of agent whose reward is computed
    :param reward_type: name of the reward function
    :param env: Environment to compute the reward function on
    :param illegal_action: True: Agent as performed an illegal action,
                           False: Agent didn't perform an illegal action
    :return: reward
    """
    if reward_type == REWARD_TYPE_PROTOCOL_REVENUE:
        return protocol_revenue(env, agent_id, illegal_action)
    elif reward_type == REWARD_TYPE_MAXIMUM_EXPOSURE:
        return maximum_exposure(env, agent_id, illegal_action)
    elif reward_type == REWARD_TYPE_PROFIT:
        return profit(env, agent_id, illegal_action)
    else:
        raise NotImplementedError("Reward function {} is unknown".format(reward_type))


def protocol_revenue(
    env,
    agent_id: int,
    illegal_action: bool
) -> float:
    """
    Function calculates the lending protocol's revenue by computing the revenue of each plf_pool
    Additionally, if a lending protocol has a negative reserve -> punishment just like illegal action

    plf_pool_revenue = plf_reserve[t] * token_price[t] - plf_reserve[t-1] * token_price[t-1]
    where, plf_reserve[t] = supply_token[t] - borrow_token[t] = available_funds[t]
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION

    lending_protocol = env.get_protocol_of_owner(agent_id)
    assert lending_protocol.owner == agent_id, f"Agent {agent_id} is not owner of the lending protocol"

    return sum([plf_pool.get_revenue() if plf_pool.reserve > 0 else REWARD_ILLEGAL_ACTION
                for plf_pool in lending_protocol.plf_pools])


def maximum_exposure(
    env,
    agent_id: int,
    illegal_action: bool
) -> float:
    """
    Function computes the maximum exposure of an agent towards a lending protocol.
    The maximum exposure can be understood as the total value of all the borrowed funds.
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION

    total_exposure = 0.0
    for lending_protocol in env.lending_protocol:
        for agent_id, pool_collateral, pool_loan in list(filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
            for borrow_hash, _ in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
                total_exposure += lending_protocol.plf_pools[pool_loan].get_borrow(borrow_hash) * lending_protocol.plf_pools[pool_loan].get_token_price()
    return total_exposure


def profit(
    env,
    agent_id: int,
    illegal_action: bool,
) -> float:
    """
    Function rewards the profit of an agent.
    profit = balance[t] - balance[t-1]
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION

    diff = 0.0
    for token_name, current in env.agent_balance[agent_id].items():
        diff += (current - env.previous_agent_balance[agent_id].get(token_name, 0)) * env.market.get_token(token_name).get_price()
    return diff
