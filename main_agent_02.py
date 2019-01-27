from run_agents import agent_02_eval_single, agent_02_eval_multiple, agent_02_eval_zero_shot

world_folder = "gen_games/twcc_easy_level5_gamesize10_step1_seed1_train-v0"

try:
    agent_02_eval_single(world_folder)
except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
