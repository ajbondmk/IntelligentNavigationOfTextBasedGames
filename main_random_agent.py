from random_agent import RandomAgent
from run_agent import test_random_agent


""" Test RandomAgent. """

agent = RandomAgent()

world_folder = "gen_games/easy/pairs/twcc_easy_level5_gamesize50_step1_seed1_test-v0"

try:
    test_random_agent(agent, world_folder, max_moves=100)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
