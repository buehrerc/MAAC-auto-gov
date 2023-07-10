import numpy as np
from itertools import product
from gym import spaces

# 1) CONFIG CONVENTIONS
CONFIG_LENDING_PROTOCOL = "lending_protocol"
CONFIG_MARKET = "market"
CONFIG_TOKEN = "token"
CONFIG_PLF_POOL = "plf_pool"
CONFIG_LP_NAME = "name"
CONFIG_PARAM = "parameter"
CONFIG_AGENT = "agent"
CONFIG_AGENT_TYPE = "type"
CONFIG_AGENT_TYPE_GOVERNANCE = "governance"
CONFIG_AGENT_TYPE_USER = "user"
CONFIG_AGENT_PROTOCOL = "protocol"
CONFIG_AGENT_BALANCE = "balance"
CONFIG_AGENT_REWARD = "reward"

# 2) ACTION ENCODING
ACTION_USER_DEPOSIT = "deposit"
ACTION_USER_WITHDRAW = "withdraw"
ACTION_USER_BORROW = "borrow"
ACTION_USER_REPAY = "repay"
ACTION_USER_LIQUIDATE = "liquidate"
ACTION_GOVERNANCE_LOWER_COLLATERAL = "lower_collateral"
ACTION_GOVERNANCE_RAISE_COLLATERAL = "raise_collateral"
ACTION_GOVERNANCE_RAISE_SLOPE_1 = "raise_borrow_slope_1"
ACTION_GOVERNANCE_LOWER_SLOPE_1 = "lower_borrow_slope_1"
ACTION_GOVERNANCE_RAISE_SLOPE_2 = "raise_borrow_slope_2"
ACTION_GOVERNANCE_LOWER_SLOPE_2 = "lower_borrow_slope_2"

# 3) CONSTANTS
# 3.1) PLFPool Constants
PLF_REWARD_RANGE = [-np.inf, np.inf]
PLF_INITIATOR = "initiator"
PLF_COLLATERAL_FACTOR_CHANGE_RATE = 0.025
PLF_INTEREST_CHANGE_RATE = 0.01
PLF_STEP_SIZE = 100
PLF_RB_FACTOR = 20
PLF_SPREAD = 0.2
PLF_FEE = 0.3
PLF_OPTIMAL_UTILIZATION_RATIO = 0.8
PLF_STABLE_BORROW_SLOPE_1 = 0.04
PLF_STABLE_BORROW_SLOPE_2 = 0.5
PLF_BASE_BORROW_RATE = 0.01
PLF_VARIABLE_BORROW_SLOPE_1 = 0.16
PLF_VARIABLE_BORROW_SLOPE_2 = 0.6

# 3.2) Lending Protocol Constants
LP_BORROW_SAFETY_MARGIN = 0.2
LP_DEFAULT_HEALTH_FACTOR = 2
LP_LIQUIDATION_PENALTY = 0.1
LP_DEPOSIT_AMOUNT = 10

# 3.3) Learning Framework Constants
EXPLORATION_RATE_1 = 1
EXPLORATION_RATE_2 = 0.1

# 4) Gym Constants
# 4.1) PLF Pool
PLF_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf]),
    dtype=np.float32
)
PLF_GOVERNANCE_ACTION_MAPPING = {
    1: ACTION_GOVERNANCE_LOWER_COLLATERAL,
    2: ACTION_GOVERNANCE_RAISE_COLLATERAL,
    3: ACTION_GOVERNANCE_LOWER_SLOPE_1,
    4: ACTION_GOVERNANCE_RAISE_SLOPE_1,
    5: ACTION_GOVERNANCE_LOWER_SLOPE_2,
    6: ACTION_GOVERNANCE_RAISE_SLOPE_2,
}
PLF_USER_ACTION_MAPPING = {
    1: ACTION_USER_DEPOSIT,
    2: ACTION_USER_WITHDRAW,
    3: ACTION_USER_BORROW,
    4: ACTION_USER_REPAY,
    5: ACTION_USER_LIQUIDATE,
}
PLF_ACTION_SPACE = {
    CONFIG_AGENT_TYPE_GOVERNANCE: len(PLF_GOVERNANCE_ACTION_MAPPING),
    CONFIG_AGENT_TYPE_USER: len(PLF_USER_ACTION_MAPPING)
}
PLF_STATES = lambda plf_number: [
    f"pool_{plf_number}/total_supply_token",
    f"pool_{plf_number}/total_borrow_token",
    f"pool_{plf_number}/reserve",
    f"pool_{plf_number}/utilization_ratio",
    f"pool_{plf_number}/collateral_factor",
    f"pool_{plf_number}/supply_interest_rate",
    f"pool_{plf_number}/borrow_interest_rate",
    f"pool_{plf_number}/base_borrow_rate",
    f"pool_{plf_number}/optimal_utilization_ratio",
    f"pool_{plf_number}/stable_borrow_slope_1",
    f"pool_{plf_number}/stable_borrow_slope_2",
]

# 4.2) LendingProtocol
LP_OBSERVATION_SPACE_1 = spaces.Box(
    low=np.array([0]),
    high=np.array([LP_DEFAULT_HEALTH_FACTOR]),
    dtype=np.float32
)
LP_OBSERVATION_SPACE_2 = spaces.Box(
    low=np.array([0]),
    high=np.array([np.inf]),
    dtype=np.float32
)
LP_STATE_RECORD = lambda agent_id, num_plf_pools: sum([
    [f"pool_{j}/agent_{agent_id}_supply" for j in range(num_plf_pools)],
    [f"pool_{j}/agent_{agent_id}_pool_{k}_borrow"
     for j, k in product(range(num_plf_pools), range(num_plf_pools)) if j != k],
], [])
LP_STATES = lambda num_plf_pools, num_agent: sum([
    sum([PLF_STATES(i) for i in range(num_plf_pools)], []),
    [f"pool_{i}/worst_loan" for i in range(num_plf_pools)],
    sum([LP_STATE_RECORD(i, num_plf_pools) for i in range(num_agent)], [])
], [])

# 4.3) Token
TOKEN_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
    dtype=np.float32
)
TOKEN_STATES = lambda token_number: [
    f"token_{token_number}/price",
    f"token_{token_number}/borrow_interest_rate",
    f"token_{token_number}/supply_interest_rate",
    f"token_{token_number}/collateral_factor",
    f"token_{token_number}/asset_volatility",
]

# 4.4) Market
MARKET_STATES = lambda token_num: sum([
    TOKEN_STATES(i) for i in range(token_num)
], [])

# 4.5) Agents
AGENT_OBSERVATION_SPACE = lambda num_tokens: spaces.Box(low=np.array([-np.inf] * num_tokens),
                                                        high=np.array([np.inf] * num_tokens),
                                                        dtype=np.float32)
AGENT_STATES = lambda agent_number, token_num: [
    f"agent{agent_number}/token_{i}_balance"
    for i in range(token_num)
]

# 4.6) Environment
ENVIRONMENT_STATES = lambda agent_num, token_num, plf_in_lp_num: sum([
    [f"lp_{lp_num}_" + state for lp_num, plf_num in enumerate(plf_in_lp_num) for state in LP_STATES(plf_num, agent_num)],
    MARKET_STATES(token_num),
    sum([AGENT_STATES(i, token_num) for i in range(agent_num)], [])
], [])

# 5) Reward Function Constants
REWARD_ILLEGAL_ACTION = -10000
REWARD_TYPE_PROTOCOL_REVENUE = "protocol_revenue"
REWARD_TYPE_MAXIMAL_BORROW_EXPOSURE = "maximal_borrow_exposure"
REWARD_TYPE_BORROW_INTEREST_RATE = "borrow_interest_rate"
REWARD_TYPE_PROFIT = "profit"
REWARD_TYPE_SUPPLY_EXPOSURE = "supply_exposure"
REWARD_TYPE_OPPORTUNITY_SUPPLY = "supply_opportunity_cost"
REWARD_TYPE_BORROW_EXPOSURE = "borrow_exposure"
REWARD_TYPE_OPPORTUNITY_BORROW = "borrow_opportunity_cost"

REWARD_TYPE_MAXIMUM_EXPOSURE = "maximum_exposure"
REWARD_TYPE_OPPORTUNITY_COST = "opportunity_cost"
REWARD_TYPE_OPPORTUNITY_SUPPLY_EXPOSURE = "opportunity_cost_supply_exposure"
REWARD_TYPE_OPPORTUNITY_BORROW_EXPOSURE = "opportunity_cost_borrow_exposure"

REWARD_CONSTANT_OPPORTUNITY_ALPHA = 1
REWARD_CONSTANT_OPPORTUNITY_BETA = 0.1
REWARD_CONSTANT_SUPPLY_LP_ID = 0
REWARD_CONSTANT_SUPPLY_PLF_ID = 1
