from typing import List, Tuple

from envs.market_env.constants import (
    REWARD_TYPE_PROTOCOL_REVENUE,
    REWARD_TYPE_MAXIMUM_EXPOSURE,
    REWARD_TYPE_PROFIT,
    REWARD_ILLEGAL_ACTION,
    REWARD_TYPE_OPPORTUNITY_COST,
    REWARD_TYPE_SUPPLY_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_SUPPLY_EXPOSURE,
    REWARD_TYPE_BORROW_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_BORROW_EXPOSURE,
    REWARD_CONSTANT_OPPORTUNITY_ALPHA,
    REWARD_CONSTANT_OPPORTUNITY_BETA,
    REWARD_CONSTANT_SUPPLY_LP_ID,
    REWARD_CONSTANT_SUPPLY_PLF_ID,
    PLF_STEP_SIZE
)


def reward_function(
    agent_id: int,
    agent_action: Tuple,
    reward_type: List[Tuple],
    env,
    illegal_action: bool,
) -> float:
    """

    Supported reward_types:
        + protocol_revenue
        + maximum_exposure
        + profit

    :param agent_id: Id of agent whose reward is computed
    :param agent_action: Action of the agent
    :param reward_type: List of reward functions: [[weight, reward_type]+]
    :param env: Environment to compute the reward function on
    :param illegal_action: True: Agent as performed an illegal action,
                           False: Agent didn't perform an illegal action
    :return: reward
    """
    # If an illegal action was picked, the agent gets a punishment
    if illegal_action:
        return REWARD_ILLEGAL_ACTION

    return sum([
        weight * reward_function_by_type(agent_id, agent_action, rt, env) for weight, rt in reward_type
    ])


def reward_function_by_type(
    agent_id: int,
    agent_action: Tuple,
    reward_type: str,
    env,
) -> float:
    """
    Supported reward_types:
        + protocol_revenue
        + maximum_exposure
        + profit
        + opportunity_cost

    :param agent_id: id of agent whose reward is computed
    :param agent_action: Action of the agent
    :param reward_type: name of the reward function
    :param env: Environment to compute the reward function on
    :return: reward
    """
    if reward_type == REWARD_TYPE_PROTOCOL_REVENUE:
        return protocol_revenue(env, agent_id)
    elif reward_type == REWARD_TYPE_MAXIMUM_EXPOSURE:
        return maximum_exposure(env, agent_id)
    elif reward_type == REWARD_TYPE_PROFIT:
        return profit(env, agent_id)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_COST:
        return opportunity_cost(env, agent_id)
    elif reward_type == REWARD_TYPE_SUPPLY_EXPOSURE:
        return supply_exposure(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_SUPPLY_EXPOSURE:
        return opportunity_cost_supply_exposure(env, agent_id)
    elif reward_type == REWARD_TYPE_BORROW_EXPOSURE:
        return borrow_exposure(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_BORROW_EXPOSURE:
        return opportunity_cost_borrow_exposure(env, agent_id)
    else:
        raise NotImplementedError("Reward function {} is unknown".format(reward_type))


def protocol_revenue(
    env,
    agent_id: int,
) -> float:
    """
    Function calculates the lending protocol's revenue by computing the revenue of each plf_pool
    Additionally, if a lending protocol has a negative reserve -> punishment just like illegal action

    plf_pool_revenue = plf_reserve[t] * token_price[t] - plf_reserve[t-1] * token_price[t-1]
    where, plf_reserve[t] = supply_token[t] - borrow_token[t] = available_funds[t]
    """
    lending_protocol = env.get_protocol_of_owner(agent_id)
    assert lending_protocol.owner == agent_id, f"Agent {agent_id} is not owner of the lending protocol"

    return sum([plf_pool.get_revenue() for plf_pool in lending_protocol.plf_pools])


def maximum_exposure(
    env,
    agent_id: int,
) -> float:
    """
    Function computes the maximum exposure of an agent towards a lending protocol.
    The maximum exposure can be understood as the total value of all the borrowed funds.
    """
    total_exposure = 0.0
    for lending_protocol in env.lending_protocol:
        for agent_id, pool_collateral, pool_loan in list(filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
            for borrow_hash, _ in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
                total_exposure += lending_protocol.plf_pools[pool_loan].get_borrow(borrow_hash) * lending_protocol.plf_pools[pool_loan].get_token_price()
    return total_exposure


def profit(
    env,
    agent_id: int,
) -> float:
    """
    Function rewards the profit of an agent.
    profit = balance[t] - balance[t-1]
    """
    diff = 0.0
    for token_name, current in env.agent_balance[agent_id].items():
        diff += (current - env.previous_agent_balance[agent_id].get(token_name, 0)) * env.market.get_token(token_name).get_price()
    return diff


def opportunity_cost(
    env,
    agent_id: int,
    alpha: float = REWARD_CONSTANT_OPPORTUNITY_ALPHA,
    beta: float = REWARD_CONSTANT_OPPORTUNITY_BETA
) -> float:
    """
    Function computes the opportunity cost of the agent's investments.
    It incorporates the opportunity cost of the supplied tokens and borrowed tokens
    based on the interest rate and collateral factor
    """
    best_supply_pool = max(
        [pool.supply_interest_rate for lp in env.lending_protocol for pool in lp.plf_pools] +
        [token.get_supply_interest_rate() for token in env.market.tokens.values()]
    )
    borrow_pools = (
        [(pool.borrow_interest_rate, pool.collateral_factor)
         for lp in env.lending_protocol for pool in lp.plf_pools] +
        [(token.get_borrow_interest_rate(), token.get_collateral_factor())
         for token in env.market.tokens.values()]
    )
    borrow_pools_ratios = [alpha * bir + beta * 1/cf for bir, cf in borrow_pools]
    best_borrow_pool = borrow_pools[borrow_pools_ratios.index(min(borrow_pools_ratios))]

    opportunity_value = 0
    for lending_protocol in env.lending_protocol:
        # Check for supply opportunity cost
        for agent_id, pool_supply in list(filter(lambda x: x[0] == agent_id, lending_protocol.supply_record)):
            for supply_hash, supply_amount in lending_protocol.supply_record[(agent_id, pool_supply)]:
                # supply_amount = lending_protocol.plf_pools[pool_supply].get_supply(supply_hash)
                supply_price = lending_protocol.plf_pools[pool_supply].get_token_price()
                opportunity_ratio = lending_protocol.plf_pools[pool_supply].supply_interest_rate - best_supply_pool
                opportunity_value += supply_amount * supply_price * opportunity_ratio
        # Check for borrow opportunity cost
        for agent_id, pool_collateral, pool_loan in list(filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
            for borrow_hash, borrow_amount in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
                # borrow_amount = lending_protocol.plf_pools[pool_collateral].get_borrow(borrow_hash)
                borrow_price = lending_protocol.plf_pools[pool_collateral].get_token_price()
                ratio = 0
                if lending_protocol.plf_pools[pool_collateral].collateral_factor - best_borrow_pool[1] != 0:
                    ratio = 1/(lending_protocol.plf_pools[pool_collateral].collateral_factor - best_borrow_pool[1])
                opportunity_ratio = alpha * (best_borrow_pool[0] - lending_protocol.plf_pools[pool_collateral].borrow_interest_rate) + \
                                    beta * ratio
                opportunity_value += borrow_amount * borrow_price * opportunity_ratio
    return opportunity_value


def supply_exposure(
    env,
    agent_id: int,
    agent_action: Tuple,
    lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
    plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific supply pool of a specific protocol
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    _, idx_lp, idx_from, idx_to = agent_action

    # Reward is positive, if the agent deposits funds into correct pool
    if lending_protocol_id == idx_lp and plf_pool_id == idx_to:
        plf_pool = env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id]
        return PLF_STEP_SIZE * plf_pool.get_token_price()
    return REWARD_ILLEGAL_ACTION / 10


def opportunity_cost_supply_exposure(
    env,
    agent_id: int,
    lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
    plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific supply pool of a specific protocol
    provided that the supply interest is lower than the competing supply interest
    """
    best_interest_rate = max(
        [lp.plf_pools[plf_pool_id].supply_interest_rate for lp in env.lending_protocol] +
        [env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id].token.get_supply_interest_rate()]
    )
    exposure = 0.0
    lending_protocol = env.lending_protocol[lending_protocol_id]
    for supply_hash, supply_amount in lending_protocol.supply_record.get((agent_id, plf_pool_id), []):
        plf_pool = lending_protocol.plf_pools[plf_pool_id]
        opportunity_diff = plf_pool.supply_interest_rate - best_interest_rate
        # If the picked lending pool offers the best interest rate -> use borrow exposure instead
        if opportunity_diff == 0:
            opportunity_diff = 1
        exposure += plf_pool.get_supply(supply_hash) * plf_pool.get_token_price() * opportunity_diff
    return exposure


def borrow_exposure(
    env,
    agent_id: int,
    agent_action: Tuple,
    lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
    plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific borrow pool of a specific protocol
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    _, idx_lp, idx_from, idx_to = agent_action

    # Reward is positive, if the agent deposits funds into correct pool
    if lending_protocol_id == idx_lp and plf_pool_id == idx_from:
        lending_protocol = env.lending_protocol[lending_protocol_id]
        borrow_hash, _ = lending_protocol.borrow_record[agent_id, idx_to, idx_from][-1]
        plf_pool = lending_protocol.plf_pools[plf_pool_id]
        return plf_pool.get_borrow(borrow_hash) * plf_pool.get_token_price()
    return REWARD_ILLEGAL_ACTION / 10


def opportunity_cost_borrow_exposure(
    env,
    agent_id: int,
    lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
    plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific borrow pool of a specific protocol
    provided that the supply interest is lower than the competing supply interest
    """
    # TODO: current implementation does not correspond to the definition in the report
    #       => the collateral factor is missing
    best_interest_rate = max(
        [lp.plf_pools[plf_pool_id].borrow_interest_rate for lp in env.lending_protocol] +
        [env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id].token.get_borrow_interest_rate()]
    )
    exposure = 0.0
    lending_protocol = env.lending_protocol[lending_protocol_id]
    for borrow_key in list(filter(lambda keys: keys[0] == agent_id and keys[2] == plf_pool_id,
                                  lending_protocol.borrow_record)):
        for borrow_hash, _ in lending_protocol.borrow_record[borrow_key]:
            plf_pool = lending_protocol.plf_pools[plf_pool_id]
            opportunity_diff = plf_pool.borrow_interest_rate - best_interest_rate
            # If the picked lending pool offers the best interest rate -> use borrow exposure instead
            if opportunity_diff == 0:
                opportunity_diff = 1
            exposure += plf_pool.get_borrow(borrow_hash) * plf_pool.get_token_price() * opportunity_diff
    return exposure
