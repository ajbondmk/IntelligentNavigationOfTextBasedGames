import os
import numpy as np
import textworld
from debug_print import debug_print, debug_not_print


def run_agent(is_training, is_random_agent, agent, world_folder, max_moves, num_episodes):
    """
    Runs an agent on all games in a world.

    Parameters:
        is_training - True if training, False if testing
        is_random_agent - True if agent is a RandomAgent, False if it is an Agent_02
        agent - the agent being run
        world_folder - the folder for the world containing games to run the agent on
        max_moves - the maximum number of moves taken before the game is failed
        num_episodes - the number of times each game is played
    """

    print()
    if is_training:
        print("TRAINING")
    else:
        print("TESTING")
    print()

    if not is_random_agent and not is_training:
        # Set the value of epsilon for testing.
        agent.set_epsilon(0.1)

    # Find all games in the chosen world.
    envs = extract_games(world_folder)
    
    # Number of games in the world.
    num_games = len(envs)

    # Initialise the arrays of move counts and scores.
    num_moves, scores = [], []

    # Repeat each game multiple times.
    for episode in range(num_episodes):

        # Run each game in the world.
        for game in range(num_games):

            # Create a TextWorld environment for the game.
            env = textworld.start(world_folder + "/" + envs[game])
            
            debug_print()
            debug_print("Game {:d}/{:d} - Episode {:d}/{:d}".format(game+1, num_games, episode+1, num_episodes))
            
            # Reset the agent and environment.
            game_state = env.reset()
            
            if is_training:

                # Update epsilon.
                if num_episodes * num_games == 1:
                    agent.set_epsilon(1)
                else:
                    agent.set_epsilon(1 - (1 - 0.02) * ((episode * (game + 1) + game) / (num_episodes * num_games - 1)))

                # Initialise the list of room descriptions seen so far.
                state_after = game_state.description
                inputs_seen = [state_after]

            for num_step in range(max_moves):
            
                # Perform the action chosen by the agent.
                action = agent.act(game_state)
                game_state, coin_reward, done = env.step(action)
                debug_print("Action:   {:s}".format(action))
                
                if is_training:
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

                # If the game is finished (completed), break.
                if done:
                    break

            # Keep track of the latest statistics.
            num_moves.append(game_state.nb_moves)
            debug_not_print("Game {:d}/{:d} - Episode {:d}/{:d} - Moves {:d}".format(game+1, num_games, episode+1, num_episodes, game_state.nb_moves))
            scores.append(game_state.score)

            debug_print()

            # Close the TextWorld environment.
            env.close()

    # Print the final statistics.
    print()
    print("Moves:", *num_moves)
    print("Average moves: {:.1f}".format(np.mean(num_moves)))
    print("Average score: {:.3f}".format(np.mean(scores)))


# Select world folder and locate its games.
def extract_games(world_folder):
    envs = []
    for filename in os.listdir(world_folder):
        if filename.endswith(".ulx"):
            envs.append(filename)
    return envs


def test_random_agent(agent, world_folder, max_moves):
    run_agent(False, True, agent, world_folder, max_moves, num_episodes=1)

def train_agent_02(agent, world_folder, max_moves, num_episodes):
    run_agent(True, False, agent, world_folder, max_moves, num_episodes)

def test_agent_02(agent, world_folder, max_moves):
    run_agent(False, False, agent, world_folder, max_moves, num_episodes=1)
