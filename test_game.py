import os
import numpy as np
import textworld
from agents.agent_02 import Agent02

game_folder = "gen_games/twcc_easy_level2_gamesize10_step1_seed1_train-v0"
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
        
        done = False
        state_after = game_state.description
        inputs_seen = [state_after]

        for num_step in range(max_moves):
            action = agent.act(game_state)
            game_state, coin_reward, done = env.step(action)
            reward = coin_reward * 10
            state_before = state_after
            state_after = game_state.description
            if reward is 0:
                if state_after not in inputs_seen:
                    reward += 1
                    inputs_seen.append(state_after)
            print(reward)
            print()
            agent.memory.add_item(state_before, action, state_after, reward)
            agent.optimise()
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
