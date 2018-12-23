import os
import numpy as np
import textworld
from agents.agent_02 import Agent02

game_folder = "gen_games/twcc_easy_level1_gamesize10_step1_seed1_train-v0"
envs = []
for filename in os.listdir(game_folder):
    if filename.endswith(".ulx"):
        envs.append(filename)

agent = Agent02()

# max_moves = 20
# num_repeats = 5
# num_games = len(envs)
max_moves = 5       # Temporary
num_repeats = 1    # Temporary
num_games = 1       # Temporary
num_moves, scores = [], []

for game in range(num_games):
    env = textworld.start(game_folder + "/" + envs[game])
    
    for episode in range(num_repeats):
        agent.reset(env)
        game_state = env.reset()
        
        # reward = 0
        done = False
        for num_step in range(max_moves):
            # command = agent.act(game_state, reward, done)
            command = agent.act(game_state)
            # game_state, reward, done = env.step(command)
            game_state, _, done = env.step(command)
            # agent.finish_step()
            if done:
                break

        num_moves.append(game_state.nb_moves)
        scores.append(game_state.score)

        # agent.finish_episode(game_state, done)

    env.close()
    print("Game {:d}/{:d} over.".format(game+1, num_games))

print()
print("Moves:", *num_moves)
print("Average moves: {:.1f}".format(np.mean(num_moves)))
print("Average score: {:.1f} / 1.0".format(np.mean(scores)))
