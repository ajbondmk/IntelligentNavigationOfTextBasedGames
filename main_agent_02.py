from agent_02 import Agent02
from run_agent import train_agent_02, test_agent_02


""" Train and test Agent02. """

agent = Agent02()

easy = "gen_games/easy/twcc_easy_level2_gamesize10_step1_seed1_train-v0"
train_world_folder = "gen_games/easy/pairs/twcc_easy_level5_gamesize500_step1_seed1_train-v0"
test_world_folder = "gen_games/easy/pairs/twcc_easy_level5_gamesize50_step1_seed1_test-v0"

train_agent_02(agent, easy, max_moves=3, num_epochs=1000, num_games=1)
print()
test_agent_02(agent, easy, max_moves=100, num_games=1)
test_agent_02(agent, easy, max_moves=100, num_games=1)
test_agent_02(agent, easy, max_moves=100, num_games=1)
test_agent_02(agent, easy, max_moves=100, num_games=1)
test_agent_02(agent, easy, max_moves=100, num_games=1)
