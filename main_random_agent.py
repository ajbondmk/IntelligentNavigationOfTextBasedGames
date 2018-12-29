from random_agent import RandomAgent
from run_agent import test_random_agent


""" Test RandomAgent. """

agent = RandomAgent()

world_folder = "gen_games/easy/pairs/twcc_easy_level3_gamesize10_step1_seed1_test-v0"

max_moves = 100
num_episodes = 1

test_random_agent(agent, world_folder, max_moves, num_episodes)
