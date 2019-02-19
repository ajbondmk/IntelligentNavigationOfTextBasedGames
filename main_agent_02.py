from run_agents import agent_02_eval_single, agent_02_eval_multiple, agent_02_eval_zero_shot

world_folder = "tw_games/level_115_games_10"

try:
    agent_02_eval_single(world_folder)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
