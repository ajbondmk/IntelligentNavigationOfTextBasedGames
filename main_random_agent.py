from random_agent import RandomAgent
from run_agents import random_agent_eval

agent = RandomAgent()

world_folder = "gen_games/easy/pairs/twcc_easy_level5_gamesize50_step1_seed1_test-v0"

try:
    random_agent_eval(agent, world_folder)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
