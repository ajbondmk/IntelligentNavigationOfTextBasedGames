from agent_02 import Agent02
from run_agents import agent_02_eval_single, agent_02_eval_multiple, agent_02_eval_zero_shot

agent = Agent02()

world_folder = "gen_games/hard/twcc_hard_level10_gamesize10_step1_seed1_train-v0"

try:
    agent_02_eval_single(agent, world_folder)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
