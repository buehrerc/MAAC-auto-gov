import numpy as np
from gym import spaces

# 1) CONFIG CONVENTIONS
CONFIG_LENDING_PROTOCOL = "lending_protocol"
CONFIG_MARKET = "market"
CONFIG_TOKEN = "token"
CONFIG_PLF_POOL = "plf_pool"
CONFIG_AGENT = "agent"
CONFIG_AGENT_TYPE = "type"
CONFIG_AGENT_TYPE_GOVERNANCE = "governance"
CONFIG_AGENT_TYPE_USER = "user"

# 2) CONSTANTS
# 2.1) PLFPool Constants
PLF_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, -np.inf, 0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
    dtype=np.float32
)
PLF_GOVERNANCE_ACTION_MAPPING = {
    1: -1,
    2: 1,
}
PLF_USER_ACTION_MAPPING = {
    1: "deposit",
    2: "withdraw",
    3: "borrow",
    4: "repay",
    5: "liquidate",
}
PLF_ACTION_SPACE = {
    CONFIG_AGENT_TYPE_GOVERNANCE: len(PLF_GOVERNANCE_ACTION_MAPPING),
    CONFIG_AGENT_TYPE_USER: len(PLF_USER_ACTION_MAPPING)
}
PLF_REWARD_RANGE = [-np.inf, np.inf]
INITIATOR = "initiator"
COLLATERAL_FACTOR_CHANGE_RATE = 0.025
PLF_STEP_SIZE = 1000
PLF_RB_FACTOR = 20
PLF_SPREAD = 0.2

# 2.2) Lending Protocol Constants
LP_BORROW_SAFETY_MARGIN = 0.15
LP_LIQUIDATION_PENALTY = 0.1
LP_DEPOSIT_AMOUNT = 1000
LP_OBSERVATION_SPACE = spaces.Box(
    low=np.array([-1]),
    high=np.array([np.inf]),
    dtype=np.float32
)

# 2.3) Token Constants
TOKEN_OBSERVATION_SPACE = spaces.Box(
    low=np.array([0, 0, 0, 0]),
    high=np.array([np.inf, np.inf, np.inf, np.inf]),
    dtype=np.float32
)