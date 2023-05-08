import json
from unittest import TestCase

from envs.market_env.market import Market
from envs.market_env.lending_protocol import LendingProtocol

CONFIG_FILE_PATH = r"./test_config/config_test.json"


class TestLendingProtocol(TestCase):
    def setUp(self) -> None:
        fs = open(CONFIG_FILE_PATH)
        self.config = json.load(fs)
        self.market = Market(config=self.config)
        self.lending_protocol = LendingProtocol(market=self.market, config=self.config)

    def test_initialization(self):
        pass




