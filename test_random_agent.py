import os
import numpy as np
import textworld
from random_agent import RandomAgent
from debug_print import debug_print, debug_not_print


""" This script tests RandomAgent on all games in a world and outputs relevant statistics. """


# Select world folder and locate its games.
world_folder = "gen_games/twcc_easy_level3_gamesize10_step1_seed1_train-v0-v0"
envs = []
for filename in os.listdir(world_folder):
    if filename.endswith(".ulx"):
        envs.append(filename)

# Maximum number of moves taken before game is failed.
max_moves = 50

# Number of games in the world.
num_games = len(envs)

# Number of times each game is played.
num_repeats = 3

# Initialise the arrays of move counts and scores.
num_moves, scores = [], []

# Create the agent.
agent = RandomAgent()


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

        for num_step in range(max_moves):
        
            # Perform the action chosen by the agent.
            action = agent.act(game_state)
            game_state, _, done = env.step(action)
            debug_print(action)

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
print("Average score: {:.3f}".format(np.mean(scores)))
