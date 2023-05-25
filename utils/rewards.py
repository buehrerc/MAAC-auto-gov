from envs.market_env.lending_protocol import LendingProtocol
from envs.market_env.constants import (
    REWARD_TYPE_PROTOCOL_REVENUE,
    REWARD_TYPE_MAXIMUM_EXPOSURE,
    REWARD_ILLEGAL_ACTION
)


def reward_function(
    agent_id: int,
    reward_type: str,
    lending_protocol: LendingProtocol,
    illegal_action: bool
) -> float:
    """
    Supported reward_types:
        + protocol_revenue
        + maximum_exposure

    :param agent_id: Id of agent whose reward is computed
    :param reward_type: name of the reward function
    :param lending_protocol: Lending protocol to compute the reward function on
    :param illegal_action: True: Agent as performed an illegal action,
                           False: Agent didn't perform an illegal action
    :return: reward
    """
    if reward_type == REWARD_TYPE_PROTOCOL_REVENUE:
        return protocol_revenue(agent_id, lending_protocol, illegal_action)
    elif reward_type == REWARD_TYPE_MAXIMUM_EXPOSURE:
        return maximum_exposure(agent_id, lending_protocol, illegal_action)
    else:
        raise NotImplementedError("Reward function {} is unknown".format(reward_type))


def protocol_revenue(
    agent_id: int,
    lending_protocol: LendingProtocol,
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

    assert lending_protocol.owner == agent_id, f"Agent {agent_id} is not owner of the lending protocol"

    return sum([plf_pool.get_revenue() if plf_pool.reserve > 0 else REWARD_ILLEGAL_ACTION
                for plf_pool in lending_protocol.plf_pools])


def maximum_exposure(
    agent_id: int,
    lending_protocol: LendingProtocol,
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
    for agent_id, pool_collateral, pool_loan in list(filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
        for borrow_hash, _ in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
            total_exposure += lending_protocol.plf_pools[pool_loan].get_borrow(borrow_hash) * lending_protocol.plf_pools[pool_loan].get_token_price()
    return total_exposure
