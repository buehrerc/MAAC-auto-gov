import json
from unittest import TestCase

from envs.market_env.market import Market

CONFIG_FILE_PATH = r"./test_config/config_test.json"


class TestMarket(TestCase):
    def setUp(self) -> None:
        fs = open(CONFIG_FILE_PATH)
        self.config = json.load(fs)
        self.market = Market(config=self.config)

    def test_initial_prices(self):
        price_dict = self.market.get_prices()
        self.assertEqual(price_dict['testToken'], 42.0)
        self.assertEqual(price_dict['anotherTestToken'], 69.0)

    def test_step(self):
        self.market.step(None)

        price_dict = self.market.get_prices()
        self.assertNotEqual(price_dict['testToken'], 42.0)
        self.assertNotEqual(price_dict['anotherTestToken'], 69.0)

    def test_reset(self):
        self.market.step(None)
        self.market.reset()

        price_dict = self.market.get_prices()
        self.assertEqual(price_dict['testToken'], 42.0)
        self.assertEqual(price_dict['anotherTestToken'], 69.0)



