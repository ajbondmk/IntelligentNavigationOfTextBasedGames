# Intelligent Navigation of Text-Based Games


## Generating Games

To generate a world (a set of games), run a command matching the following format:

`tw-make tw-coin_collector --level {level} --output tw_games/{folder_name}/`

where `level` is defined as follows:

|   `level`  |  mode  |  quest length |     total rooms     |  distractor rooms  |
|:----------:|:------:|:-------------:|:-------------------:|:------------------:|
|  1 to 100  |  easy  |    `level`    |       `level`       |        none        |
| 101 to 200 | medium | `level` % 100 | 2 * (`level` % 100) | one per chain room |
| 201 to 300 |  hard  | `level` % 100 | 3 * (`level` % 100) | two per chain room |
<!-- TODO: Is this true for hard mode distractor rooms? -->


## Agents

`RandomAgent` selects actions randomly, ignoring text generated by the game.

`Agent02` uses a neural network with an LSTM layer and DQN for training. Actions are selected using an epsilon-greedy policy, where epsilon begins at 1 decreases linearly until it ends at 0.


## Training and Testing

A script which tests `RandomAgent` can be found in `my_code/main_random_agent.py`.

A script which trains and tests `Agent02` can be found in `my_code/main_agent_02.py`.

These script for `Agent02` can be edited to point to run one of the following tests:
- `agent_02_eval_single`
- `agent_02_eval_multiple`
- `agent_02_eval_zero_shot`
<!-- TODO: Add descriptions. -->

Both scripts (for `RandomAgent` and `Agent02`) can be edited to point to different world folders.
