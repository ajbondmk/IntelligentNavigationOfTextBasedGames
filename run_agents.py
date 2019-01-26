import os
import numpy as np
import textworld


# The maximum number of moves taken before the game is failed.
MAX_MOVES_TRAIN = 50
MAX_MOVES_TEST = 200

# The number of times each game is repeated.
NUM_EPOCHS_AGENT_02_TRAIN = 1000
NUM_EPOCHS_RANDOM_TEST = 100

# The number of epochs between training the model.
TRAIN_INTERVAL = 4

# The number of epochs between testing the agent (during training).
TEST_INTERVAL = 20


def train_and_test_agent_02(agent, envs, test_envs, results_file):
    """
    Trains an Agent02 on a set of games, testing periodically.

    Parameters:
        agent - the agent being run
        envs - the list of environments to train on
        test_envs - the list of environments to test on ([] means test on training envs)
        results_file - the file to which results should be written
    """

    print()

    # Epsilon will decay from 1 to epsilon_limit_value over the first epsilon_limit_epoch epochs. then remain at epsilon_limit_value.
    epsilon_limit_value = 0.2
    epsilon_limit_epoch = 1000

    # Number of different games to be played.
    num_games = len(envs)

    # If no test environments are passed in, test on the training environments.
    if test_envs == []:
        test_envs = envs

    # Initialise the arrays of move counts and scores.
    num_moves, scores = [], []

    # Initialse the global step count to 0.
    count = 0
    
    # Write a header to the results file.
    with open(results_file, "a") as f:
        f.write("\n\nTRAINING ON GAMES:\n")
        for env in envs:
            f.write("    {}\n".format(env))

    # Repeat each game multiple times.
    for epoch in range(NUM_EPOCHS_AGENT_02_TRAIN):

        # Run each game in the set of input games.
        for game in range(num_games):

            # Create a TextWorld environment for the game.
            env = textworld.start(envs[game])
            
            # Reset the agent and environment.
            game_state = env.reset()

            # Update epsilon.
            if epoch < epsilon_limit_epoch - 1:
                agent.set_epsilon(1 - ((1 - epsilon_limit_value) * (epoch / (epsilon_limit_epoch - 2))))
            elif epoch == epsilon_limit_epoch - 1:
                agent.set_epsilon(epsilon_limit_value)

            # Initialise the list of room descriptions seen so far.
            state_after = game_state.description
            inputs_seen = [state_after]

            # Allow the agent to perform up to MAX_MOVES_TRAIN actions.
            for _ in range(MAX_MOVES_TRAIN):
            
                # Perform the action chosen by the agent.
                action = agent.act(game_state)
                game_state, reward, done = env.step(action)
                
                # Update the local state variables.
                state_before = state_after
                state_after = game_state.description

                # Give a reward of 1 point if the agent picked up the coin or discovered a new room, and 0 points otherwise.
                if reward is 0:
                    if state_after not in inputs_seen:
                        reward += 1
                        inputs_seen.append(state_after)
            
                # Add the transition to memory.
                agent.memory.add_item(state_before, action, state_after, reward)

                # Every TRAIN_INTERVAL steps, train the agent.
                count = count + 1
                if count % TRAIN_INTERVAL == 0:
                    agent.optimise()

                # If the game is finished (completed), break.
                if done:
                    break

            # Keep track of the statistics for this game and epoch, printing to console.
            num_moves.append(game_state.nb_moves)
            scores.append(game_state.score)
            print("Game {:d}/{:d} - Epoch {:d}/{:d} - Moves {:d}".format(game+1, num_games, epoch+1, NUM_EPOCHS_AGENT_02_TRAIN, game_state.nb_moves))

            # Close the TextWorld environment.
            env.close()

        # Every TEST_INTERVAL epochs, test the agent.
        if (epoch+1) % TEST_INTERVAL == 0:
            with open(results_file, "a") as f:
                f.write("{}   ".format(epoch+1))
            test_agent_02(agent=agent, envs=test_envs, results_file=results_file)
    
    print()


def test_agent_02(agent, envs, results_file):
    """
    Tests an Agent02 on a set of games.

    Parameters:
        agent - the agent being run
        envs - the list of environments
        results_file - the file to which results should be written
    """

    # Set the value of epsilon for testing.
    agent.set_epsilon(0)

    # Number of games to be played.
    num_games = len(envs)

    # Initialise the arrays of move counts and scores.
    num_moves, scores = [], []

    # Run each game in the set of input games.
    for game in range(num_games):

        # Create a TextWorld environment for the game.
        env = textworld.start(envs[game])
        
        # Reset the agent and environment.
        game_state = env.reset()

        # Allow the agent to perform up to MAX_MOVES_TEST actions.
        for _ in range(MAX_MOVES_TEST):
        
            # Perform the action chosen by the agent.
            action = agent.act(game_state)
            game_state, _, done = env.step(action)

            # If the game is finished (completed), break.
            if done:
                break

        # Keep track of the latest statistics.
        num_moves.append(game_state.nb_moves)
        scores.append(game_state.score)

        # Close the TextWorld environment.
        env.close()

    # Print statistics (to console and results_file).
    with open(results_file, "a") as f:
        f.write("{:.1f}   {:.1f}\n".format(np.mean(num_moves), np.mean(scores)))
    print("\nAverage moves: {:.1f}, Average score: {:.1f}\n".format(np.mean(num_moves), np.mean(scores)))


def test_random_agent(agent, envs, results_file):
    """
    Tests a RandomAgent on a set of games.

    Parameters:
        agent - the agent being run
        envs - the list of environments
        results_file - the file to which results should be written
    """

    print()

    # Number of games to be played.
    num_games = len(envs)

    # Initialise the arrays of move counts and scores.
    num_moves, scores = [], []

    # Run each game in the world.
    for game in range(num_games):

        # Repeat each game multiple times.
        for epoch in range(NUM_EPOCHS_RANDOM_TEST):

            # Create a TextWorld environment for the game.
            env = textworld.start(envs[game])
            
            # Reset the agent and environment.
            game_state = env.reset()

            # Allow the agent to perform up to MAX_MOVES_TEST actions.
            for _ in range(MAX_MOVES_TEST):
            
                # Perform the action chosen by the agent.
                action = agent.act(game_state)
                game_state, _, done = env.step(action)

                # If the game is finished (completed), break.
                if done:
                    break

            # Keep track of the latest statistics, printing to console.
            num_moves.append(game_state.nb_moves)
            scores.append(game_state.score)
            print("Game {:d}/{:d} - Epoch {:d}/{:d} - Moves {:d}".format(game+1, num_games, epoch+1, NUM_EPOCHS_RANDOM_TEST, game_state.nb_moves))

            # Close the TextWorld environment.
            env.close()

        # Print test statistics (to console and results_file).
        with open(results_file, "a") as f:
            f.write("{}   {:.1f}   {:.1f}\n".format(game, np.mean(num_moves), np.mean(scores)))
        print()
        print("Game: {}, Average moves: {:.1f}, Average score: {:.1f}".format(game, np.mean(num_moves), np.mean(scores)))
        print()
    
    print()


def extract_games(world_folder):
    """ Locate the games in a world folder. """
    envs = []
    for filename in os.listdir(world_folder):
        if filename.endswith(".ulx"):
            envs.append(world_folder + "/" + filename)
    return envs

def generate_results_file_name():
    """ Generate a filename to output results to, based on the current time and date. """
    return "test_results/{}_{}.txt".format(datetime.now().date(), datetime.now().time())


def random_agent_eval(agent, world_folder):
    #TODO: Add description.
    envs = extract_games(world_folder)
    results_file = generate_results_file_name()
    test_random_agent(
        agent=agent,
        envs=envs,
        results_file=results_file)


def agent_02_eval_single(agent, world_folder):
    # TODO: Add description.
    envs = extract_games(world_folder)
    results_file = generate_results_file_name()
    for env in envs:
        train_and_test_agent_02(
            agent=agent,
            envs=[env],
            test_envs=[],
            results_file=results_file
        )

def agent_02_eval_multiple(agent, world_folder):
    # TODO: Add description.
    envs = extract_games(world_folder)
    results_file = generate_results_file_name()
    train_and_test_agent_02(
        agent=agent,
        envs=envs,
        test_envs=[],
        results_file=results_file
    )

def agent_02_eval_zero_shot(agent, train_world_folder, test_world_folder):
    # TODO: Add description.
    train_envs = extract_games(train_world_folder)
    test_envs = extract_games(test_world_folder)
    results_file = generate_results_file_name()
    train_and_test_agent_02(
        agent=agent,
        envs=train_envs,
        test_envs=test_envs,
        results_file=results_file
    )
