import os
import numpy as np
import textworld
from agents.agent_02 import Agent02
from agents.debug_print import debug_print
from agents.debug_print import debug_not_print

world_folder = "gen_games/twcc_easy_level3_gamesize10_step1_seed1_train-v0"
envs = []
for filename in os.listdir(world_folder):
    if filename.endswith(".ulx"):
        envs.append(filename)

max_moves = 50
num_games = len(envs)
num_repeats = 10
num_moves, scores = [], []

agent = Agent02(num_games * num_repeats)

for episode in range(num_repeats):

    for game in range(num_games):

        env = textworld.start(world_folder + "/" + envs[game])
        
        debug_print()
        debug_print("Game {:d}/{:d} - Episode {:d}/{:d}".format(game, num_games-1, episode, num_repeats-1))
        
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
            debug_print("Reward:   {:d}".format(reward))
            
            agent.memory.add_item(state_before, action, state_after, reward)
            agent.optimise()
            if done:
                break

        num_moves.append(game_state.nb_moves)
        debug_not_print("Game {:d}/{:d} - Episode {:d}/{:d} - Moves {:d}".format(game, num_games-1, episode, num_repeats-1, game_state.nb_moves))
        scores.append(game_state.score)

        debug_print()

        env.close()

print()
print("Moves:", *num_moves)
print("Average moves: {:.1f}".format(np.mean(num_moves)))
print("Average score: {:.1f} / 1.0".format(np.mean(scores)))
