import numpy as np
from gym import spaces

# 1) CONFIG CONVENTIONS
CONFIG_LENDING_PROTOCOL = "lending_protocol"
CONFIG_MARKET = "market"
CONFIG_TOKEN = "token"
CONFIG_PLF_POOL = "plf_pool"
CONFIG_PARAM = "parameter"
CONFIG_AGENT = "agent"
CONFIG_AGENT_TYPE = "type"
CONFIG_AGENT_TYPE_GOVERNANCE = "governance"
CONFIG_AGENT_TYPE_USER = "user"
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
PLF_STEP_SIZE = 1000
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
LP_BORROW_SAFETY_MARGIN = 0.15
LP_DEFAULT_HEALTH_FACTOR = 2
LP_LIQUIDATION_PENALTY = 0.1
LP_DEPOSIT_AMOUNT = 10

# 4) Gym Constants
# 4.1) PLF Pool
PLF_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, -np.inf, 0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
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

# 4.2) LendingProtocol
LP_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0]),
    high=np.array([LP_DEFAULT_HEALTH_FACTOR]),
    dtype=np.float32
)

# 4.3) Token
TOKEN_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf]),
    dtype=np.float32
)

# 4.4) Agents
AGENT_OBSERVATION_SPACE = lambda num_plf_pools: spaces.Box(low=np.array([-np.inf]*num_plf_pools),
                                                           high=np.array([np.inf]*num_plf_pools),
                                                           dtype=np.float32)

# 5) Reward Function Constants
REWARD_ILLEGAL_ACTION = -10000
REWARD_TYPE_PROTOCOL_REVENUE = "protocol_revenue"
REWARD_TYPE_MAXIMUM_EXPOSURE = "maximum_exposure"

