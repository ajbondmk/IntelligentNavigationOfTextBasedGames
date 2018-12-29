# Intelligent Navigation of Text-Based Games


## Generating Games

To generate a world (a set of games), run a command matching the following format:

`tw-make.py twcc_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)`

where:
- `mode` is easy if there are no distractor nodes and hard if there are a lot.
- `level` is the number of steps needed to solve each game.
- `gamesize` is the number of games in the world.
- `step` is ???.
- `seed` is the random seed.
- `split` determines which group of data these games belong to.

The new world folder will likely be located in a folder named `gen_games`.


## Agents

`RandomAgent` selects actions randomly, ignoring text generated by the game.

`Agent02` uses a neural network with an LSTM layer and DQN for training. Actions are selected using an epsilon-greedy policy, where epsilon begins at 1 decreases linearly until it ends at 0.


## Training and Testing

A script which tests `RandomAgent` can be found in `my_code/main_random_agent.py`.

A script which trains and tests `Agent02` can be found in `my_code/main_agent_02.py`.

The parameters `max_moves` and `num_episodes` can be edited in these files. For `RandomAgent`, the parameter `world_folder` can be set; for `Agent02`, the parameters `train_world_folder` and `test_world_folder` can bbe set.

To provide a more verbose terminal output, open the file `debug_print.py` and set `DEBUG` to `True`. For a less verbose output, set it to `False`.
