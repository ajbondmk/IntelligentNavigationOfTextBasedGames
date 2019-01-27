from run_agents import random_agent_eval

world_folder = "gen_games/twcc_easy_level5_gamesize10_step1_seed1_test-v0"

try:
    random_agent_eval(world_folder)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
