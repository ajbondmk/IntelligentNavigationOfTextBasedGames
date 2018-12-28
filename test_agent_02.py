import os
import numpy as np
import textworld
from agent_02 import Agent02
from debug_print import debug_print, debug_not_print


""" This script trains Agent02 on all games in a world and outputs relevant statistics. """


# Select world folder and locate its games.
world_folder = "gen_games/twcc_easy_level3_gamesize10_step1_seed1_train-v0"
envs = []
for filename in os.listdir(world_folder):
    if filename.endswith(".ulx"):
        envs.append(filename)

# Maximum number of moves taken before game is failed.
max_moves = 50

# Number of games in the world.
num_games = len(envs)

# Number of times each game is repeated.
num_repeats = 10

# Initialise the arrays of move counts and scores.
num_moves, scores = [], []

# Create the agent.
agent = Agent02(num_games * num_repeats)


# Repeat each game multiple times.
for episode in range(num_repeats):

    # Run each game in the world.
    for game in range(num_games):

        # Create a TextWorld environment for the game.
        env = textworld.start(world_folder + "/" + envs[game])
        
        debug_print()
        debug_print("Game {:d}/{:d} - Episode {:d}/{:d}".format(game, num_games-1, episode, num_repeats-1))
        
        # Reset the agent and environment.
        agent.reset(env)
        game_state = env.reset()
        
        # Initialise the list of room descriptions seen so far.
        state_after = game_state.description
        inputs_seen = [state_after]

        for num_step in range(max_moves):
        
            # Perform the action chosen by the agent.
            action = agent.act(game_state)
            game_state, coin_reward, done = env.step(action)
            
            state_before = state_after
            state_after = game_state.description

            # Calculate the reward:
            #    10 points if the agent picked up the coin.
            #    1 point if the agent discovered a new room.
            #    0 points otherwise.
            reward = coin_reward * 10
            if reward is 0:
                if state_after not in inputs_seen:
                    reward += 1
                    inputs_seen.append(state_after)
            debug_print("Reward:   {:d}".format(reward))
            
            # Add the transition to memory.
            agent.memory.add_item(state_before, action, state_after, reward)

            # Train the agent.
            agent.optimise()

            # If the game is finished (completed or failed), break.
            if done:
                break

        # Keep track of the latest statistics.
        num_moves.append(game_state.nb_moves)
        debug_not_print("Game {:d}/{:d} - Episode {:d}/{:d} - Moves {:d}".format(game, num_games-1, episode, num_repeats-1, game_state.nb_moves))
        scores.append(game_state.score)

        debug_print()

        # Close the TextWorld environment.
        env.close()


# Print the final statistics.
print()
print("Moves:", *num_moves)
print("Average moves: {:.1f}".format(np.mean(num_moves)))
print("Average score: {:.1f} / 1.0".format(np.mean(scores)))
