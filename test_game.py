import numpy as np
import textworld
from agents.random_agent import RandomAgent

env = textworld.start("gen_games/twcc_easy_level2_gamesize1_step1_seed1_train-v0/twcc_easy_level2_gamesize1_step1_seed1_train-v0_Z9mDDVCXI9ZvioBxuXWrI9mWfxQP.ulx")
agent = RandomAgent()

max_moves = 100
num_episodes = 10
num_moves, scores = [], []

for episode in range(num_episodes):
    agent.reset(env)
    game_state = env.reset()
    
    reward = 0
    done = False
    for no_step in range(max_moves):
        command = agent.act(game_state, reward, done)
        game_state, reward, done = env.step(command)
        if done:
            break

    num_moves.append(game_state.nb_moves)
    scores.append(game_state.score)

env.close()

print("Moves:", *num_moves)
print("Average moves: {:.1f}".format(np.mean(num_moves)))
print("Average score: {:.1f} / 1.0".format(np.mean(scores)))