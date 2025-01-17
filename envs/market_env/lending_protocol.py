import logging
import uuid
import torch
import numpy as np
from itertools import product
from gym import spaces
from gym.core import ObsType, RenderFrame
from typing import Dict, List, Tuple, Optional, Union

from envs.market_env.market import Market
from envs.market_env.plf_pool import PLFPool
from envs.market_env.utils import (
    combine_observation_space,
)
from envs.market_env.constants import (
    CONFIG_AGENT,
    CONFIG_AGENT_TYPE,
    LP_BORROW_SAFETY_MARGIN,
    LP_LIQUIDATION_PENALTY,
    LP_OBSERVATION_SPACE_1,
    LP_OBSERVATION_SPACE_2,
    LP_DEFAULT_HEALTH_FACTOR
)


class LendingProtocol:
    """
    Class implements an AAVE-like over-collateralized lending protocol.
    """

    def __init__(
        self,
        name: str,
        owner: int,
        market: Market,
        plf_pool: List[Dict],
        config: Dict,
        agent_balance: List[Dict]
    ):
        self.name = name
        self.owner = owner
        self.market = market
        self.plf_pool_config = plf_pool
        self.config = config

        # Protocol Book-keeping
        self.supply_record: Dict[Tuple, List] = dict()  # dict_keys: (agent_id, pool_id)
        self.borrow_record: Dict[Tuple, List] = dict()  # dict_keys: (agent_id, pool_collateral, pool_loan)
        self.worst_loans: Dict[int, Tuple] = dict()     # dict_keys: pool_id, dict_values: (borrow_keys, loan_hash, health_factor)
        self.reward: torch.Tensor = torch.Tensor()

        # Initialize additional attributes
        self.plf_pools: List[PLFPool] = list()
        self.agent_mask: List[str] = [agent[CONFIG_AGENT_TYPE] for agent in self.config[CONFIG_AGENT]]
        self.agent_balance = agent_balance

        # Gym Environment Attributes
        self.observation_space = spaces.Space()
        self.agent_action_space = list()

        self.reset(new_agent_balance=self.agent_balance)

# =====================================================================================================================
#   ENVIRONMENT ACTIONS
# =====================================================================================================================
    def reset(self, new_agent_balance: List[Dict]) -> ObsType:
        """
        Resets the LendingProtocol and all its plf pools to the initial parameters by reinitializing the PLFPools
        :return: new_observation
        """
        self.agent_balance = new_agent_balance
        # (Re-)Initialize the plf pools
        self.plf_pools = list(
            map(
                lambda params: PLFPool(market=self.market, owner=self.owner, agent_balance=self.agent_balance, **params),
                self.plf_pool_config
            )
        )
        # Based on the reinitialized plf pools -> compute observation and action space
        num_plf_pols = len(self.plf_pools)
        # Worst loan observation space
        additional_observation_spaces = [LP_OBSERVATION_SPACE_1] * num_plf_pols
        # observation space for supply and borrow balance of each agent
        additional_observation_spaces.extend(
            [LP_OBSERVATION_SPACE_2] * len(self.agent_mask) * (num_plf_pols + num_plf_pols * (num_plf_pols - 1))
        )
        self.observation_space = combine_observation_space(self.plf_pools, additional_observation_spaces)

        # Reset the reward
        self.reward = torch.zeros(len(self.agent_mask))

        # Reset all the records
        self.supply_record = dict()
        self.borrow_record = dict()

        # Retrieve the state of the lending protocol by updating it
        return self.update()

    def update(self) -> ObsType:
        """
        Function updates the lending protocol's state.
        1) Updates all pools
        2) Updates the loan records
        3) Computes the rewards
        :return: new_observations
        """
        # 1) Update all plf_pools based on the actions of the agents
        pool_states = [pool.update() for pool in self.plf_pools]

        # 2) Compute the lowest health factor for each plf_pool
        self.worst_loans = {i: self._update_health_factor(i) for i in range(len(self.plf_pools))}

        # Append them to the state
        pool_states.extend([
            torch.Tensor([v[2]]) if v is not None else torch.Tensor([LP_DEFAULT_HEALTH_FACTOR])
            for v in self.worst_loans.values()
        ])

        # 3) Append the supplies and borrows of each agent as states
        pool_states.extend([self._get_agent_records(agent_id) for agent_id in range(len(self.agent_mask))])

        return torch.cat(pool_states)

    def _get_health_factor(self, pool_collateral: int, pool_loan: int, loan_hash: str) -> float:
        """
        The health factor of a loan can be computed as follows:

        H = (collateral_factor * loan_amount * loan_price) / (loan_amount * loan_price)
        """
        loan_amount = self.plf_pools[pool_loan].get_borrow(loan_hash)
        loan_price = self.plf_pools[pool_loan].get_token_price()
        collateral_amount = self.plf_pools[pool_collateral].get_supply(loan_hash)
        collateral_price = self.plf_pools[pool_collateral].get_token_price()
        collateral_factor = self.plf_pools[pool_loan].get_collateral_factor()
        return (collateral_amount * collateral_price * collateral_factor) / (loan_amount * loan_price)

    def _update_health_factor(self, idx: int):
        """
        Function retrieves the loan of each plf pool with the lowest health factor
        """
        loan_in_pool = list(filter(lambda x: x[2] == idx, self.borrow_record.keys()))
        if len(loan_in_pool) == 0:
            return None
        # Compute health_factor for all loans
        unpacked_pool_loans = [(*k, loans) for k in loan_in_pool for loans, _ in self.borrow_record[k]]
        if len(unpacked_pool_loans) == 0:
            return None
        health_factors = list(map(lambda x: self._get_health_factor(*x[1:]), unpacked_pool_loans))
        min_hf_idx = np.argmin(health_factors)
        return unpacked_pool_loans[min_hf_idx][:3], unpacked_pool_loans[min_hf_idx][3], health_factors[min_hf_idx]

    def _get_agent_records(self, agent_id: int) -> torch.Tensor:
        """
        Function returns the aggregate of all the supplied and borrowed tokens of the input agent_id
        """
        record = list()
        # 1) Supplied tokens
        for pool_supply in range(len(self.plf_pools)):
            record.append(sum(list(map(lambda key: key[1], self.supply_record.get((agent_id, pool_supply), [])))))
        # 2) Borrowed tokens
        for pool_loan, pool_collateral in product(range(len(self.plf_pools)), range(len(self.plf_pools))):
            if pool_loan == pool_collateral:
                continue
            record.append(sum(list(map(lambda key: key[1], self.borrow_record.get((agent_id, pool_collateral, pool_loan), [])))))
        return torch.Tensor(record)

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def get_name(self) -> str:
        return self.name

# =====================================================================================================================
#   AGENT ACTIONS
# =====================================================================================================================
    def update_collateral_factor(self, pool_id: int, direction: int) -> bool:
        """
        Function updates the collateral factor of the corresponding plf_pool
        :param pool_id: id of the plf_pool whose collateral factor is to be updated
        :param direction: -1=decrease, +1=increase

        :return: True: illegal_action, False: legal_action
        """
        return self.plf_pools[pool_id].update_collateral_factor(direction)

    def update_interest_model(self, pool_id: int, stable_borrow_slope_1: int = 0, stable_borrow_slope_2: int = 0) -> bool:
        """
        Function updates the interest model's parameters.
        For now, it's the two slopes of the stable borrow interest model

        :return: True: illegal_action, False: legal_action
        """
        return self.plf_pools[pool_id].update_interest_model(stable_borrow_slope_1, stable_borrow_slope_2)

    def deposit(self, agent_id: int, pool_to: int, amount: float) -> bool:
        """
        Function deposits funds into the corresponding plf_pool
        :param agent_id: id of agent who wants to deposit funds
        :param pool_to: id of pool to which funds are deposited
        :param amount: amount of funds which are to be deposited

        :return: True: illegal_action, False: legal_action
        """
        # 1) Deduct the funds from the agents balance
        feedback = self._remove_agent_funds(agent_id, pool_to, amount)
        if feedback:  # Agent doesn't have enough funds -> illegal action
            return feedback

        # 2) Record the deposit
        deposit_hash = uuid.uuid4().hex
        if self.supply_record.get((agent_id, pool_to)) is None:
            self.supply_record[agent_id, pool_to] = list()
        self.supply_record[agent_id, pool_to].append((deposit_hash, amount))

        # 3) Add the funds to the pool
        self.plf_pools[pool_to].add_supply(key=deposit_hash, amount=amount)

        return False

    def withdraw(self, agent_id: int, pool_from: int, amount: float) -> bool:
        """
        Function withdraws funds from the corresponding plf_pool
        :param agent_id: id of agent who wants to withdraw funds
        :param pool_from: id of pool from which funds are withdrawn
        :param amount: amount of funds which are to be withdrawn

        :return: True: illegal_action, False: legal_action
        """
        pool_token = self.plf_pools[pool_from].get_token_name()

        # 1) Check whether the user has deposited anything into the pool
        if self.supply_record.get((agent_id, pool_from)) is None or len(self.supply_record.get((agent_id, pool_from))) == 0:
            logging.debug(
                f"Agent {agent_id} tried to withdraw {amount} from {pool_token}, "
                f"but didn't deposit enough or didn't deposit at all."
            )
            return True

        # 2) Get the hash value of the supply
        withdraw_hash, initial_amount = self.supply_record.get((agent_id, pool_from)).pop()

        # 3) Remove the funds from the pool
        withdraw_amount = self.plf_pools[pool_from].remove_supply(withdraw_hash)

        # 4) Add the funds to the agent's and protocol owner's balance
        self.agent_balance[agent_id][pool_token] += withdraw_amount

        return False

    def borrow(self, agent_id: int, pool_collateral: int, pool_loan: int, amount: float) -> (float, bool):
        """
        The agent borrows funds with a safety margin which is set as a constant (LP_BORROW_SAFETY_MARGIN)
        borrow_amount = (deposit_amount * deposit_price * (borrow_collateral_factor + LP_BORROW_SAFETY_MARGIN)) / borrow_price

        :param agent_id: ID of agent who tries to borrow
        :param pool_collateral: PLFPool, to which the collateral will be deposited
        :param pool_loan: PLFPool, from which the funds are borrowed
        :param amount: Amount of funds that are being deposited

        :return: True: illegal_action, False: legal_action
        """
        # 1) Compute the deposit and borrow amount
        deposit_amount = amount
        deposit_price = self.plf_pools[pool_collateral].get_token_price()
        borrow_price = self.plf_pools[pool_loan].get_token_price()
        l_t = self.plf_pools[pool_loan].get_collateral_factor() - LP_BORROW_SAFETY_MARGIN
        borrow_amount = (deposit_amount * deposit_price * l_t) / borrow_price
        if borrow_amount < 0:
            # The borrow_amount can become lower than zero, if the collateral factor is lower than
            # the borrow safety margin
            borrow_amount = 0

        # 2) Deduct the collateral from the agent first
        feedback = self._remove_agent_funds(agent_id, pool_collateral, deposit_amount)
        # Agent doesn't have enough funds
        if feedback:
            logging.debug(
                f"Agent {agent_id} tried to borrow funds from pool {pool_loan} by "
                f"providing collateral to pool {pool_collateral}, but didn't have enough funds."
            )
            return feedback

        # 3) Record the borrow
        loan_hash = uuid.uuid4().hex
        if self.borrow_record.get((agent_id, pool_collateral, pool_loan)) is None:
            self.borrow_record[agent_id, pool_collateral, pool_loan] = list()
        self.borrow_record[agent_id, pool_collateral, pool_loan].append((loan_hash, deposit_amount))

        # 4) Deposit the collateral (2nd transaction for step 2)
        self.plf_pools[pool_collateral].add_supply(key=loan_hash, amount=deposit_amount)

        # 4) Withdraw the borrowed funds
        borrow_token = self.plf_pools[pool_loan].get_token_name()
        self.agent_balance[agent_id][borrow_token] += borrow_amount
        self.plf_pools[pool_loan].start_borrow(key=loan_hash, amount=borrow_amount)

        return False

    def repay(self, agent_id: int, pool_loan: int, pool_collateral: int, amount: float) -> bool:
        """
        :param agent_id: ID of agent who tries to repay the loan
        :param pool_loan: PLFPool, to which loan is repaid
        :param pool_collateral: PLFPool, from which the collateral is repaid
        :param amount: Amount of funds that are being deposited

        :return: True: illegal_action, False: legal_action
        """
        # 1) Check whether an according borrow exists
        if self.borrow_record.get((agent_id, pool_collateral, pool_loan)) is None or \
                len(self.borrow_record.get((agent_id, pool_collateral, pool_loan))) == 0:
            logging.debug(
                f"Agent {agent_id} tried to repay a loan, but never borrowed funds from pool {pool_loan}."
            )
            return True

        # 2) Retrieve the loan hash
        loan_hash, initial_amount = self.borrow_record[(agent_id, pool_collateral, pool_loan)][0]

        # 3) Agent pays the borrowed funds
        borrowed_amount = self.plf_pools[pool_loan].return_borrow(loan_hash)
        feedback = self._remove_agent_funds(agent_id, pool_loan, borrowed_amount)
        if feedback:
            # Agent cannot repay the borrowed funds -> reset the borrowed funds in the pool
            self.plf_pools[pool_loan].start_borrow(loan_hash, borrowed_amount)
            logging.debug(
                f"Agent {agent_id} tried to repay the loan from pool {pool_loan}, "
                f"but didn't have enough funds."
            )
            return True

        # 4) Transfer the collateral back to the agent
        collateral_token = self.plf_pools[pool_collateral].get_token_name()
        collateral_amount = self.plf_pools[pool_collateral].remove_supply(loan_hash)
        self.agent_balance[agent_id][collateral_token] += collateral_amount

        # 5) Remove the borrow record
        self.borrow_record[(agent_id, pool_collateral, pool_loan)].pop(0)

        return False

    def liquidate(self, agent_id: int, pool_id: int) -> bool:
        """
        Function liquidates the unhealthiest loan in the corresponding plf_pool.

        :param agent_id: id of agent who wants to liquidate a loan
        :param pool_id: id of pool which is meant to be liquidated

        :return: True: illegal_action, False: legal_action
        """
        # 1) Check whether the pool has an unhealthy loan
        if self.worst_loans.get(pool_id) is None:
            return True

        borrow_key, loan_hash, health_factor = self.worst_loans.get(pool_id)
        if health_factor > 1:
            logging.debug(
                f"Agent {agent_id} tried to liquidate a loan from pool {pool_id},"
                f" however the loan has a good health factor: {health_factor}"
            )
            return True

        # 2) Repay the loan funds
        liquidated_agent_id, pool_collateral, pool_loan = borrow_key
        loan_amount = self.plf_pools[pool_loan].get_borrow(loan_hash)
        if loan_amount is None:
            logging.debug(
                f"Agent {agent_id} tried to liquidate pool {pool_loan} "
                f"but another agent has already liquidated the loan."
            )
            # Since this is per se not an illegal action -> we leave it as a valid action without an effect
            return False

        # 3) Remove funds from the agent
        feedback = self._remove_agent_funds(agent_id, pool_loan, loan_amount)
        if feedback:
            logging.debug(
                f"Agent {agent_id} tried to liquidate pool {pool_loan} by paying {loan_amount},"
                f"however it didn't have enough funds."
            )
            return True
        # Remove the borrow entry from the pool
        self.plf_pools[pool_loan].return_borrow(loan_hash)

        # 4) Pay fees on the collateral_amount
        collateral_token = self.plf_pools[pool_collateral].get_token_name()
        loan_id = list(map(lambda x: x[0], self.borrow_record[borrow_key])).index(loan_hash)
        collateral_amount = self.plf_pools[pool_collateral].remove_supply(loan_hash)
        loan_hash, initial_collateral_amount = self.borrow_record[borrow_key][loan_id]

        # 5) Distribute the collateral to the liquidator (agent_id) and
        #    the remaining value to the liquidated agent (liquidated_agent_id)
        loan_plus_penalty = loan_amount * self.plf_pools[pool_loan].get_token_price() * (1 + LP_LIQUIDATION_PENALTY)
        collateral_value = collateral_amount * self.plf_pools[pool_collateral].get_token_price()

        # Liquidator receives loan_plus_penalty -> loan_amount
        liquidator_amount = loan_plus_penalty / self.plf_pools[pool_collateral].get_token_price()
        remaining_amount = (collateral_value - loan_plus_penalty) / self.plf_pools[pool_collateral].get_token_price()
        assert liquidator_amount > 0, "Liquidated amount has to be positive."
        self.agent_balance[agent_id][collateral_token] += liquidator_amount
        if remaining_amount > 0:
            # After the deduction of the liquidation penalty, the remaining amount still belongs to initial borrower
            self.agent_balance[liquidated_agent_id][collateral_token] += remaining_amount
        else:
            # The protocol has to absorb the losses
            if self.supply_record.get(("Losses", None)) is None:
                self.supply_record["Losses", None] = list()
            self.supply_record["Losses", None].append((None, remaining_amount))
        logging.debug(
            f"Pool {pool_id} was liquidated, liquidator paid {loan_amount} "
            f"and received {liquidator_amount}. The remaining {remaining_amount} "
            f"were transfered to Agent {liquidated_agent_id}"
         )

        # 6) Remove the loan from the borrow_record
        self.borrow_record[borrow_key].pop(loan_id)

        return False

    def _remove_agent_funds(self, agent_id: int, pool_to: int, amount: float) -> bool:
        """
        :return: True: illegal_action, False: legal_action
        """
        pool_token = self.plf_pools[pool_to].get_token_name()
        agent_balance = self.agent_balance[agent_id].get(pool_token)
        if agent_balance < amount:
            logging.debug(
                f"Agent {agent_id} tried to deposit {amount} into {pool_token},"
                f" but didn't have enough funds ({agent_balance})"
            )
            return True
        self.agent_balance[agent_id][pool_token] -= amount
        return False

    def __repr__(self):
        return f"LendingProtocol(name: '{self.name}', " + repr(self.plf_pools) + ")"
