from agent_02 import Agent02
from run_agent import train_agent_02, test_agent_02


""" Train Agent02 (and test along the way). """

agent = Agent02()

world_folder = "gen_games/easy/twcc_easy_level5_gamesize10_step1_seed1_train-v0"

try:
    train_agent_02(agent, world_folder, max_moves=50, num_epochs=1000, num_games=1)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
