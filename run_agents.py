""" File containing code to train and test agents by performing experiments. """


import os
import csv
import numpy as np
import textworld
from agent_02 import Agent02
from random_agent import RandomAgent


# The maximum number of moves taken before the game is failed.
MAX_MOVES_TRAIN = 50
MAX_MOVES_TEST = 200

# The number of times each game is repeated.
NUM_EPOCHS_AGENT_02_TRAIN = 3000
NUM_EPOCHS_RANDOM_TEST = 10

# The number of epochs between training the model.
TRAIN_INTERVAL = 4


def train_and_test_agent_02(agent, envs, test_envs_list):
    """
    Trains an Agent02 on a set of games.

    Parameters:
        agent - the agent being run
        envs - the list of environments to train on
        test_envs_list - the list of lists of environments to test on (zero shot evaluation)
    """

    # Epsilon will decay from 1 to epsilon_limit_value over the first epsilon_limit_epoch epochs,
    # then remain at epsilon_limit_value.
    epsilon_limit_value = 0.2
    epsilon_limit_epoch = 1000

    # Number of different games to be played.
    num_games = len(envs)

    # Initialse the global step count to 0.
    count = 0

    # Repeat each game multiple times.
    for epoch in range(NUM_EPOCHS_AGENT_02_TRAIN):

        num_moves, scores = [], []

        # Run each game in the set of input games.
        for game in range(num_games):

            # Create a TextWorld environment and game state for the game.
            env = textworld.start(envs[game])
            env.enable_extra_info("description")
            game_state = env.reset()

            # Update epsilon.
            if epoch < epsilon_limit_epoch - 1:
                agent.set_epsilon(
                    1 - ((1 - epsilon_limit_value) * (epoch / (epsilon_limit_epoch - 2))))
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

                # Give a reward of 1 point if the agent picked up the coin or discovered a new room,
                # and 0 points otherwise.
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

            # Close the TextWorld environment.
            env.close()

        # Add latest statistics to the agent.
        agent.num_moves_results.append(np.mean(num_moves))
        agent.score_results.append(np.mean(scores))

        # If no test environments are passed in, test on the training environments.
        if test_envs_list != []:
            test_agent_02(agent=agent, test_sets=test_envs_list)


def test_agent_02(agent, test_sets):
    """
    Tests an Agent02 on a set of games.

    Parameters:
        agent - the agent being run
        test_sets - the list of environments
    """

    # Set the value of epsilon for testing.
    agent.set_epsilon(0)

    for i in range(len(test_sets)):

        # Initialise the arrays of move counts and scores.
        num_moves, scores = [], []

        # Run each game in the set of input games.
        for j in range(len(test_sets[i])):

            # Create a TextWorld environment and game state for the game.
            env = textworld.start(test_sets[i][j])
            env.enable_extra_info("description")
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

        # Add latest statistics to the agent.
        agent.num_moves_results_test[i].append(np.mean(num_moves))
        agent.score_results_test[i].append(np.mean(scores))


def test_random_agent(agent, envs):
    """
    Tests a RandomAgent on a set of games.

    Parameters:
        agent - the agent being run
        envs - the list of environments
    """

    # Number of games to be played.
    num_games = len(envs)

    # Initialise the array of scores.
    scores = []

    # Run each game in the world.
    for game in range(num_games):

        # Repeat each game multiple times.
        for _ in range(NUM_EPOCHS_RANDOM_TEST):

            # Create a TextWorld environment and game state for the game.
            env = textworld.start(envs[game])
            env.enable_extra_info("description")
            game_state = env.reset()

            # Allow the agent to perform up to MAX_MOVES_TEST actions.
            for _ in range(MAX_MOVES_TEST):

                # Perform the action chosen by the agent.
                action = agent.act()
                game_state, _, done = env.step(action)

                # If the game is finished (completed), break.
                if done:
                    break

            # Keep track of the latest statistics.
            scores.append(game_state.score)

            # Close the TextWorld environment.
            env.close()

    # Print the average reward over all games.
    print("Average reward:", np.mean(scores))


def extract_games(world_folder):
    """ Locate the games in a world folder. """
    envs = []
    for filename in os.listdir(world_folder):
        if filename.endswith(".ulx"):
            envs.append(world_folder + "/" + filename)
    return envs


def generate_results_file_name(world_folder):
    """ Generate a filename to output results to, with no file extension. """
    name = world_folder
    if world_folder[:9] == "tw_games/":
        if world_folder[-1:] == "/":
            name = world_folder[9:-1]
        else:
            name = world_folder[9:]
    else:
        name = world_folder
    return "test_results/" + name


def output_to_csvs(agent, results_file_name):
    """ Output all training/testing data to CSV files. """
    output_to_csv(agent.num_moves_results, results_file_name + "_moves.csv")
    output_to_csv(agent.score_results, results_file_name + "_scores.csv")
    if agent.num_moves_results_test:
        for line in agent.num_moves_results_test:
            output_to_csv(line, results_file_name + "_test-moves.csv")
    if agent.score_results_test:
        for line in agent.score_results_test:
            output_to_csv(line, results_file_name + "_test-scores.csv")

def output_to_csv(results, results_file_name):
    """ Output a line to a CSV file. """
    with open(results_file_name, mode='a') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)


def random_agent_eval(world_folder):
    #TODO: Add description.
    agent = RandomAgent()
    envs = extract_games(world_folder)
    test_random_agent(
        agent=agent,
        envs=envs
    )


def agent_02_eval_single(world_folder):
    # TODO: Add description.
    envs = extract_games(world_folder)
    results_file_name = generate_results_file_name(world_folder)
    for env in envs:
        agent = Agent02()
        train_and_test_agent_02(
            agent=agent,
            envs=[env],
            test_envs_list=[]
        )
        output_to_csvs(agent, results_file_name)

def agent_02_eval_multiple(world_folder):
    # TODO: Add description.
    agent = Agent02()
    envs = extract_games(world_folder)
    results_file_name = generate_results_file_name(world_folder)
    train_and_test_agent_02(
        agent=agent,
        envs=envs,
        test_envs_list=[]
    )
    output_to_csvs(agent, results_file_name)

def agent_02_eval_zero_shot(train_world_folder, test_world_folders):
    # TODO: Add description.
    agent = Agent02()

    # train_envs = extract_games(train_world_folder)
    # test_envs_list = []
    # for i in range(test_world_folders):
    #     test_envs_list[i] = extract_games(test_world_folders[i])
    # results_file_name = generate_results_file_name(train_world_folder)
    train_envs = extract_games("tw_games/zero_shot/train/010_005")
    test_envs_list = []
    test_envs_list.append(extract_games("tw_games/zero_shot/test/005"))
    test_envs_list.append(extract_games("tw_games/zero_shot/test/015"))
    test_envs_list.append(extract_games("tw_games/zero_shot/test/025"))
    results_file_name = "test_results/zero_shot/005"
    agent.num_moves_results_test.append([])
    agent.num_moves_results_test.append([])
    agent.num_moves_results_test.append([])
    agent.score_results_test.append([])
    agent.score_results_test.append([])
    agent.score_results_test.append([])

    train_and_test_agent_02(
        agent=agent,
        envs=train_envs,
        test_envs_list=test_envs_list
    )
    output_to_csvs(agent, results_file_name)
