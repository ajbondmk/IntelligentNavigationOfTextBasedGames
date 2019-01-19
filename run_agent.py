import os
import numpy as np
import textworld


def run_agent(is_training, is_random_agent, agent, world_folder, max_moves, num_epochs, num_games):
    """
    Runs an agent on all games in a world.

    Parameters:
        is_training - True if training, False if testing
        is_random_agent - True if agent is a RandomAgent, False if it is an Agent_02
        agent - the agent being run
        world_folder - the folder for the world containing games to run the agent on
        max_moves - the maximum number of moves taken before the game is failed
        num_epochs - the number of times each game is played
        num_games = number of games from the world_folder which should be played (0 indicates all)
    """

    print()

    # Set the lower value which epsilon should reduce to over time.
    epsilon_limit_value = 0.2
    epsilon_limit_epoch = 1000

    if not is_random_agent and not is_training:
        # Set the value of epsilon for testing.
        agent.set_epsilon(epsilon_limit_value)

    # Find all games in the chosen world.
    envs = extract_games(world_folder)
    
    # Number of games in the world.
    if num_games == 0 or num_games > len(envs):
        num_games = len(envs)

    # Initialise the arrays of move counts and scores.
    num_moves, scores = [], []

    # Initialse the global step count to 0.
    count = 0

    # Repeat each game multiple times.
    for epoch in range(num_epochs):

        # Run each game in the world.
        for game in range(num_games):

            # Create a TextWorld environment for the game.
            env = textworld.start(world_folder + "/" + envs[game])
            
            # Reset the agent and environment.
            game_state = env.reset()
            
            if is_training:

                # Update epsilon.
                if epoch < epsilon_limit_epoch - 1:
                    agent.set_epsilon(1 - ((1 - epsilon_limit_value) * (epoch / (epsilon_limit_epoch - 2))))
                elif epoch == epsilon_limit_epoch - 1:
                    agent.set_epsilon(epsilon_limit_value)

                # Initialise the list of room descriptions seen so far.
                state_after = game_state.description
                inputs_seen = [state_after]

            for num_step in range(max_moves):
            
                # Perform the action chosen by the agent.
                action = agent.act(game_state)
                game_state, reward, done = env.step(action)
                
                if is_training:
                    state_before = state_after
                    state_after = game_state.description

                    # Give a reward of 1 point if the agent picked up the coin or discovered a new room, and 0 points otherwise.
                    if reward is 0:
                        if state_after not in inputs_seen:
                            reward += 1
                            inputs_seen.append(state_after)
                
                    # Add the transition to memory.
                    agent.memory.add_item(state_before, action, state_after, reward)

                    # Every 4 steps, train the agent.
                    count = count + 1
                    if count % 4 == 0:
                        agent.optimise()

                # If the game is finished (completed), break.
                if done:
                    break

            # Keep track of the latest statistics.
            num_moves.append(game_state.nb_moves)
            scores.append(game_state.score)
            if is_training:
                print("Game {:d}/{:d} - Epoch {:d}/{:d} - Moves {:d}".format(game+1, num_games, epoch+1, num_epochs, game_state.nb_moves))

            # Close the TextWorld environment.
            env.close()

        # Periodically throughout training, perform some tests.
        test_interval = 50
        if is_training:
            if (epoch+1) % test_interval == 0:
                test_agent_02(agent, world_folder, max_moves=200, num_epochs=5, num_games=num_games)

    # Print test statistics.
    if not is_training:
        with open("TestResults.txt", "a") as f:
            f.write("{:.1f}   {:.1f}\n".format(np.mean(num_moves), np.mean(scores)))
        print("Average moves: {:.1f}".format(np.mean(num_moves)))
        print("Average score: {:.1f}".format(np.mean(scores)))
    
    print()


# Select world folder and locate its games.
def extract_games(world_folder):
    envs = []
    for filename in os.listdir(world_folder):
        if filename.endswith(".ulx"):
            envs.append(filename)
    return envs


def test_random_agent(agent, world_folder, max_moves, num_epochs=1, num_games=0):
    run_agent(False, True, agent, world_folder, max_moves, num_epochs, num_games)

def train_agent_02(agent, world_folder, max_moves, num_epochs=1, num_games=0):
    run_agent(True, False, agent, world_folder, max_moves, num_epochs, num_games)

def test_agent_02(agent, world_folder, max_moves, num_epochs=1, num_games=0):
    run_agent(False, False, agent, world_folder, max_moves, num_epochs, num_games)
