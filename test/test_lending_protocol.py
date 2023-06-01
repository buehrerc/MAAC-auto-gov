import json
from unittest import TestCase

from envs.market_env.env import MultiAgentEnv
from envs.market_env.constants import (
    PLF_INTEREST_CHANGE_RATE,
    PLF_COLLATERAL_FACTOR_CHANGE_RATE,
    PLF_STABLE_BORROW_SLOPE_1,
    PLF_STABLE_BORROW_SLOPE_2,

)

CONFIG_FILE_PATH = r"./test_config/config_test.json"


class TestLendingProtocol(TestCase):
    def setUp(self) -> None:
        fs = open(CONFIG_FILE_PATH)
        self.config = json.load(fs)
        fs.close()

        self.env = MultiAgentEnv(config=self.config, seed=0)
        self.env.reset()
        self.lending_protocol = self.env.lending_protocol[0]

    def test_initialization(self):
        self.assertEqual(len(self.lending_protocol.plf_pools), 3)
        self.assertEqual(len(self.lending_protocol.agent_balance), 2)
        self.assertEqual(len(self.lending_protocol.supply_record), 0)
        self.assertEqual(len(self.lending_protocol.borrow_record), 0)

    def test_increase_collateral(self):
        initial_collateral_factor = self.lending_protocol.plf_pools[0].get_collateral_factor()
        self.assertFalse(self.lending_protocol.update_collateral_factor(0, 1))
        self.assertEqual(self.lending_protocol.plf_pools[0].get_collateral_factor(), initial_collateral_factor + PLF_COLLATERAL_FACTOR_CHANGE_RATE)

    def test_decrease_collateral(self):
        initial_collateral_factor = self.lending_protocol.plf_pools[0].get_collateral_factor()
        self.assertFalse(self.lending_protocol.update_collateral_factor(0, -1))
        self.assertEqual(self.lending_protocol.plf_pools[0].get_collateral_factor(), initial_collateral_factor - PLF_COLLATERAL_FACTOR_CHANGE_RATE)

    def test_increase_interest_rate_slope(self):
        self.assertFalse(self.lending_protocol.update_interest_model(0, 1, 1))
        self.assertEqual(self.lending_protocol.plf_pools[0].stable_borrow_slope_1, PLF_STABLE_BORROW_SLOPE_1 + PLF_INTEREST_CHANGE_RATE)
        self.assertEqual(self.lending_protocol.plf_pools[0].stable_borrow_slope_2, PLF_STABLE_BORROW_SLOPE_2 + PLF_INTEREST_CHANGE_RATE)

    def test_decrease_interest_rate_slope(self):
        self.assertFalse(self.lending_protocol.update_interest_model(0, -1, -1))
        self.assertEqual(self.lending_protocol.plf_pools[0].stable_borrow_slope_1, PLF_STABLE_BORROW_SLOPE_1 - PLF_INTEREST_CHANGE_RATE)
        self.assertEqual(self.lending_protocol.plf_pools[0].stable_borrow_slope_2, PLF_STABLE_BORROW_SLOPE_2 - PLF_INTEREST_CHANGE_RATE)

    def test_increase_collateral_unallowed(self):
        for i in range(5):
            self.assertFalse(self.lending_protocol.update_collateral_factor(0, 1))
        self.assertTrue(self.lending_protocol.update_collateral_factor(0, 1))

    def test_deposit(self):
        self.assertFalse(self.lending_protocol.deposit(1, 0, 100))
        self.assertEqual(self.lending_protocol.agent_balance[1]['TKN'], 900)
        self.assertEqual(len(self.lending_protocol.supply_record[(1, 0)]), 1)
        self.assertEqual(self.lending_protocol.supply_record[(1, 0)][0][1], 100)

    def test_deposit_not_enough_funds(self):
        self.assertTrue(self.lending_protocol.deposit(1, 0, 1e10))

    def test_withdraw(self):
        self.assertFalse(self.lending_protocol.deposit(1, 0, 100))
        self.assertFalse(self.lending_protocol.withdraw(1, 0, 100))
        self.assertEqual(self.lending_protocol.agent_balance[1]['TKN'], 1000)
        self.assertEqual(len(self.lending_protocol.supply_record[(1, 0)]), 0)

    def test_withdraw_no_deposit(self):
        self.assertTrue(self.lending_protocol.withdraw(1, 0, 1000))

    def test_borrow(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        self.assertEqual(self.lending_protocol.agent_balance[1]['TKN'], 900)
        self.assertEqual(self.lending_protocol.agent_balance[1]['USDC'], 7100)
        self.assertEqual(len(self.lending_protocol.borrow_record[(1, 0, 1)]), 1)
        self.assertEqual(self.lending_protocol.borrow_record[(1, 0, 1)][0][1], 100)

    def test_borrow_not_enough_funds(self):
        self.assertTrue(self.lending_protocol.borrow(1, 0, 1, 1e10))

    def test_repay(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        self.assertFalse(self.lending_protocol.repay(1, 1, 0, 100))
        self.assertEqual(self.lending_protocol.agent_balance[1]['TKN'], 1000)
        self.assertEqual(self.lending_protocol.agent_balance[1]['USDC'], 5000)
        self.assertEqual(len(self.lending_protocol.borrow_record[(1, 0, 1)]), 0)

    def test_repay_no_loan(self):
        self.assertTrue(self.lending_protocol.repay(1, 1, 0, 100))

    def test_repay_not_enough_funds(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        # Hack agent balance
        self.lending_protocol.agent_balance[1]['USDC'] = 0
        self.assertTrue(self.lending_protocol.repay(1, 1, 0, 100))

    def test_liquidate(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        # Hack market price to offset health factor of the loan
        self.lending_protocol.plf_pools[0].token.price = 1e-5
        self.env.step((0, 0))

        self.assertFalse(self.lending_protocol.liquidate(1, 1))

    def test_liquidate_no_loan(self):
        self.assertTrue(self.lending_protocol.liquidate(1, 0))

    def test_liquidate_healthy_loan(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        self.assertTrue(self.lending_protocol.liquidate(1, 1))

    def test_liquidate_not_enough_funds(self):
        self.assertFalse(self.lending_protocol.borrow(1, 0, 1, 100))
        # Hack the agent's balance + make the loan unhealthy
        self.lending_protocol.agent_balance[1]['USDC'] = 0
        self.lending_protocol.plf_pools[0].token.price = 1
        self.env.step((0, 0))
        self.assertTrue(self.lending_protocol.liquidate(1, 1))
