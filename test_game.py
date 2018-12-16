import os
import numpy as np
import textworld
from agents.agent_01 import Agent01

game_folder = "gen_games/twcc_easy_level2_gamesize10_step1_seed1_train-v0"
envs = []
for filename in os.listdir(game_folder):
    if filename.endswith(".ulx"):
        envs.append(filename)

agent = Agent01()

max_moves = 20
num_episodes = 1
num_games = len(envs)
num_moves, scores = [], []

for game in range(num_games):
    env = textworld.start(game_folder + "/" + envs[game])
    
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
    print("Game {:d}/{:d} over.".format(game+1, num_games))

print()
print("Moves:", *num_moves)
print("Average moves: {:.1f}".format(np.mean(num_moves)))
print("Average score: {:.1f} / 1.0".format(np.mean(scores)))
