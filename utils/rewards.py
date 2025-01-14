from typing import List, Tuple

from envs.market_env.constants import (
    REWARD_TYPE_PROTOCOL_REVENUE,
    REWARD_TYPE_MAXIMUM_EXPOSURE,
    REWARD_TYPE_MAXIMAL_BORROW_EXPOSURE,
    REWARD_TYPE_BORROW_INTEREST_RATE,
    REWARD_TYPE_PROFIT,
    REWARD_ILLEGAL_ACTION,
    REWARD_TYPE_OPPORTUNITY_COST,
    REWARD_TYPE_SUPPLY_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_SUPPLY_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_SUPPLY,
    REWARD_TYPE_BORROW_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_BORROW_EXPOSURE,
    REWARD_TYPE_OPPORTUNITY_BORROW,
    REWARD_CONSTANT_OPPORTUNITY_ALPHA,
    REWARD_CONSTANT_OPPORTUNITY_BETA,
    REWARD_CONSTANT_SUPPLY_LP_ID,
    REWARD_CONSTANT_SUPPLY_PLF_ID,
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
    elif reward_type == REWARD_TYPE_MAXIMAL_BORROW_EXPOSURE:
        return maximal_borrow_exposure(env, agent_id)
    elif reward_type == REWARD_TYPE_BORROW_INTEREST_RATE:
        return borrow_interest_rate(env, agent_id)
    elif reward_type == REWARD_TYPE_PROFIT:
        return profit(env, agent_id)
    elif reward_type == REWARD_TYPE_SUPPLY_EXPOSURE:
        return supply_exposure(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_BORROW_EXPOSURE:
        return borrow_exposure(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_SUPPLY:
        return supply_opportunity_cost(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_BORROW:
        return borrow_opportunity_cost(env, agent_id, agent_action)

    # Legacy reward functions
    elif reward_type == REWARD_TYPE_MAXIMUM_EXPOSURE:
        return maximum_exposure(env, agent_id)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_COST:
        return opportunity_cost(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_BORROW_EXPOSURE:
        return opportunity_cost_borrow_exposure(env, agent_id, agent_action)
    elif reward_type == REWARD_TYPE_OPPORTUNITY_SUPPLY_EXPOSURE:
        return opportunity_cost_supply_exposure(env, agent_id, agent_action)
    else:
        raise NotImplementedError("Reward function {} is unknown".format(reward_type))


# =====================================================================================================================
#   REWARD FUNCTIONS
# =====================================================================================================================


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


def maximal_borrow_exposure(
        env,
        agent_id: int,
) -> float:
    """
    Function computes the exposure of borrowed assets.
    The higher the exposure the higher the reward
    """
    lending_protocol = env.get_protocol_of_owner(agent_id)
    assert lending_protocol.owner == agent_id, f"Agent {agent_id} is not owner of the lending protocol"

    total_exposure = 0.0
    for keys, values in lending_protocol.borrow_record.items():
        _, _, pool_loan = keys
        for borrow_hash, _ in values:
            total_exposure += lending_protocol.plf_pools[pool_loan].get_borrow(borrow_hash) * \
                              lending_protocol.plf_pools[pool_loan].get_token_price()
    previous_exposure = env.previous_maximal_exposure if hasattr(env, "previous_maximal_exposure") else 0
    new_exposure = total_exposure - previous_exposure
    env.previous_maximal_exposure = total_exposure
    return new_exposure


def profit(
        env,
        agent_id: int,
) -> float:
    """
    Function rewards the profit of an agent.
    profit = balance[t] - balance[t-1]
    """
    diff = 0.0
    previous_balance = env.previous_agent_balance[0]
    for token_name, current in env.agent_balance[agent_id].items():
        diff += (current - previous_balance[agent_id].get(token_name, 0)) * env.market.get_token(token_name).get_price()
    return diff


def borrow_interest_rate(
        env,
        agent_id: int,
) -> float:
    lending_protocol = env.get_protocol_of_owner(agent_id)
    own_interest_rate = min([plf_pool.borrow_interest_rate for plf_pool in lending_protocol.plf_pools])
    concurrent_interest_rate = min([token.borrow_interest_rate for token in env.market.tokens.values()] +
                                   [plf_pool.borrow_interest_rate for lp in env.lending_protocol for plf_pool in lp.plf_pools])

    if own_interest_rate < concurrent_interest_rate:
        return -REWARD_ILLEGAL_ACTION
    else:
        return REWARD_ILLEGAL_ACTION


def supply_exposure(
        env,
        agent_id: int,
        agent_action: Tuple,
        lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
        plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    The function rewards the immediate action of the agent by
    rewarding exposure to a specific supply pool of a specific protocol
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    # Reward is positive, if the agent deposits funds into correct pool
    if not (action_id == 1 and
            idx_lp == lending_protocol_id and
            idx_from is None and
            idx_to == plf_pool_id):
        return REWARD_ILLEGAL_ACTION

    lending_protocol = env.lending_protocol[lending_protocol_id]
    supply_hash, _ = lending_protocol.supply_record[agent_id, idx_to][-1]
    plf_pool = env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id]
    return plf_pool.get_supply(supply_hash) * plf_pool.get_token_price()


def borrow_exposure(
        env,
        agent_id: int,
        agent_action: Tuple,
        lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
        plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    The function rewards the immediate action of the agent by
    rewarding exposure to a specific borrow pool of a specific protocol
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    # Reward is positive, if the agent deposits funds into correct pool
    if not (action_id == 3 and
            idx_lp == lending_protocol_id and
            idx_from == plf_pool_id):
        return REWARD_ILLEGAL_ACTION

    lending_protocol = env.lending_protocol[lending_protocol_id]
    borrow_hash, _ = lending_protocol.borrow_record[agent_id, idx_to, idx_from][-1]
    plf_pool = lending_protocol.plf_pools[plf_pool_id]
    return plf_pool.get_borrow(borrow_hash) * plf_pool.get_token_price()


def supply_opportunity_cost(
        env,
        agent_id: int,
        agent_action: Tuple,
) -> float:
    """
    Function centers its reward around the opportunity costs of the agent's actions.
    More specifically, it focuses on the opportunity costs which a supply action inflicts.
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    if not (action_id == 0 or action_id == 1):
        return REWARD_ILLEGAL_ACTION

    best_pool_interest_rate = max([plf_pool.supply_interest_rate
                                   for lp in env.lending_protocol for plf_pool in lp.plf_pools])
    best_market_interest_rate = max([token.supply_interest_rate for token in env.market.tokens.values()])
    best_interest_rate = max([best_pool_interest_rate, best_market_interest_rate])

    if action_id == 0:
        if best_pool_interest_rate < best_market_interest_rate:
            # If all supply interest rate are lower than the market -> do not supply
            return 100
        else:
            return REWARD_ILLEGAL_ACTION

    # Best interest rate is provided by a pool
    lending_protocol = env.lending_protocol[idx_lp]
    plf_pool = lending_protocol.plf_pools[idx_to]
    opportunity_diff = plf_pool.supply_interest_rate - best_interest_rate
    supply_hash, _ = lending_protocol.supply_record[agent_id, idx_to][-1]
    # If the picked lending pool offers the best interest rate -> use borrow exposure instead
    if opportunity_diff == 0:
        opportunity_diff = 1
    return plf_pool.get_supply(supply_hash) * plf_pool.get_token_price() * opportunity_diff


def borrow_opportunity_cost(
        env,
        agent_id: int,
        agent_action: Tuple,
) -> float:
    """
    Function centers its reward around the opportunity costs of the agent's actions.
    More specifically, it focuses on the opportunity costs which a borrowing action inflicts.
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    if not (action_id == 0 or action_id == 3):
        return REWARD_ILLEGAL_ACTION

    best_pool_interest_rate = min([plf.previous_borrow_interest_rate[0]
                                   for lp in env.lending_protocol for plf in lp.plf_pools])
    best_market_interest_rate = min([token.borrow_interest_rate for token in env.market.tokens.values()])
    best_interest_rate = min(best_pool_interest_rate, best_market_interest_rate)
    agent_interest_rate = (env.lending_protocol[idx_lp].plf_pools[idx_from].previous_borrow_interest_rate[0]
                           if action_id == 3
                           else best_market_interest_rate)
    borrow_value = 10000

    if action_id == 3:
        lending_protocol = env.lending_protocol[idx_lp]
        plf_pool = lending_protocol.plf_pools[idx_from]
        borrow_hash, _ = lending_protocol.borrow_record[agent_id, idx_to, idx_from][-1]
        borrow_value = plf_pool.get_borrow(borrow_hash) * plf_pool.get_token_price()

    opportunity_diff = best_interest_rate - agent_interest_rate
    if opportunity_diff == 0:
        opportunity_diff = 1
    else:
        opportunity_diff *= 100
    reward = borrow_value * opportunity_diff
    return reward if reward > REWARD_ILLEGAL_ACTION else REWARD_ILLEGAL_ACTION


# =====================================================================================================================
#   LEGACY REWARD FUNCTIONS
# =====================================================================================================================
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
    borrow_pools_ratios = [alpha * bir + beta * 1 / cf for bir, cf in borrow_pools]
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
        for agent_id, pool_collateral, pool_loan in list(
                filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
            for borrow_hash, borrow_amount in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
                # borrow_amount = lending_protocol.plf_pools[pool_collateral].get_borrow(borrow_hash)
                borrow_price = lending_protocol.plf_pools[pool_collateral].get_token_price()
                ratio = 0
                if lending_protocol.plf_pools[pool_collateral].collateral_factor - best_borrow_pool[1] != 0:
                    ratio = 1 / (lending_protocol.plf_pools[pool_collateral].collateral_factor - best_borrow_pool[1])
                opportunity_ratio = alpha * (
                            best_borrow_pool[0] - lending_protocol.plf_pools[pool_collateral].borrow_interest_rate) + \
                                    beta * ratio
                opportunity_value += borrow_amount * borrow_price * opportunity_ratio
    return opportunity_value


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
        for agent_id, pool_collateral, pool_loan in list(
                filter(lambda x: x[0] == agent_id, lending_protocol.borrow_record)):
            for borrow_hash, _ in lending_protocol.borrow_record[(agent_id, pool_collateral, pool_loan)]:
                total_exposure += lending_protocol.plf_pools[pool_loan].get_borrow(borrow_hash) * \
                                  lending_protocol.plf_pools[pool_loan].get_token_price()
    return total_exposure


def opportunity_cost_supply_exposure(
        env,
        agent_id: int,
        agent_action: Tuple,
        lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
        plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific supply pool of a specific protocol
    provided that the supply interest is lower than the competing supply interest
    """
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    # Reward is positive, if the agent deposits funds into correct pool
    if not ((action_id == 1 and
             idx_lp == lending_protocol_id and
             idx_from is None and
             idx_to == plf_pool_id) or (action_id == 0)):
        return REWARD_ILLEGAL_ACTION

    best_interest_rate = max(
        [lp.plf_pools[plf_pool_id].supply_interest_rate for lp in env.lending_protocol] +
        [env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id].token.get_supply_interest_rate()]
    )

    lending_protocol = env.lending_protocol[lending_protocol_id]
    plf_pool = env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id]
    opportunity_diff = plf_pool.supply_interest_rate - best_interest_rate

    # if opportunity_diff is negative (i.e. the plf pool does not offer the best interest_rate)
    # and agent chose 0 -> reward
    if action_id == 0:
        if opportunity_diff < 0:
            return 10
        else:
            return REWARD_ILLEGAL_ACTION

    supply_hash, _ = lending_protocol.supply_record[agent_id, idx_to][-1]
    # If the picked lending pool offers the best interest rate -> use borrow exposure instead
    if opportunity_diff == 0:
        opportunity_diff = 1
    return plf_pool.get_supply(supply_hash) * plf_pool.get_token_price() * opportunity_diff


def opportunity_cost_borrow_exposure(
        env,
        agent_id: int,
        agent_action: Tuple,
        lending_protocol_id: int = REWARD_CONSTANT_SUPPLY_LP_ID,
        plf_pool_id: int = REWARD_CONSTANT_SUPPLY_PLF_ID,
) -> float:
    """
    Function rewards exposure to a specific borrow pool of a specific protocol
    provided that the supply interest is lower than the competing supply interest
    """
    # TODO: current implementation does not correspond to the definition in the report
    #       => the collateral factor is missing
    assert len(agent_action) == 4, "Agent type is incorrect!"
    action_id, idx_lp, idx_from, idx_to = agent_action

    ## Reward is positive, if the agent deposits funds into correct pool
    if not ((action_id == 1 and
            idx_lp == lending_protocol_id and
            idx_from is None and
            idx_to == plf_pool_id) or (action_id == 0)):
        return REWARD_ILLEGAL_ACTION

    best_interest_rate = min(
        [lp.plf_pools[plf_pool_id].borrow_interest_rate for lp in env.lending_protocol] +
        [env.lending_protocol[lending_protocol_id].plf_pools[plf_pool_id].token.get_borrow_interest_rate()]
    )

    lending_protocol = env.lending_protocol[lending_protocol_id]
    borrow_hash, _ = lending_protocol.borrow_record[agent_id, idx_to, idx_from][-1]
    plf_pool = lending_protocol.plf_pools[plf_pool_id]
    opportunity_diff = plf_pool.borrow_interest_rate - best_interest_rate
    # If the picked lending pool offers the best interest rate -> use borrow exposure instead
    if opportunity_diff == 0:
        opportunity_diff = 1
    return plf_pool.get_borrow(borrow_hash) * plf_pool.get_token_price() * opportunity_diff
