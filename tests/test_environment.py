import unittest
from src.environment import Environment
import numpy as np
from src.models import Day_Ahead_Trader


class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = Environment()

    def test_reset(self):
        """
        make a couple of random actions in the environment, reset the environment and take a couple of steps again.
        Test if the environment behaves as expected.
        Returns:

        """
        reset_files = []
        for i in range((24+12)*4):
            action = (np.random.random(1) - 0.5) * 0.3
            next_state, reward, done, info = self.environment.step(action)
            reset_files.append(next_state["day-ahead-price"])

        self.environment.reset()
        self.assertEqual( self.environment.data_loader.current_index, 0)

        for i in range(12*4+1, (24+12)*4):
            action = (np.random.random(1) - 0.5) * 0.3
            next_state, reward, done, info = self.environment.step(action)
            #self.assertEqual(np.sum(reset_files[(12*41+1+i)]), np.sum(next_state["day-ahead-price"]))

    def test_dataloader(self):
        """
        Get dataloader from environment and iterate call get_next_day_ahead_and_intraday_price until it returns done
        Reset the dataloader afterwards and iterate again
        Returns:

        """
        dataloader = self.environment.data_loader
        done = False
        for i in range(3):
            while not done:
                day_ahead_price, intraday_price, done = dataloader.get_next_day_ahead_and_intraday_price()
                dataloader.reset()


    def test_day_ahead_trader(self):
        """
        Test the
        Returns:

        """
        env = self.environment
        trader = Day_Ahead_Trader()
        env.reset()
        action = 0
        sum_of_rewards = 0
        while True:
            next_state, reward, done, info = env.step(action)

            if done:
                break
            action = trader.step(next_state, reward)
            sum_of_rewards += reward
        self.environment.reset()
        self.assertGreater(sum_of_rewards, 100)

if __name__ == '__main__':
    unittest.main()
