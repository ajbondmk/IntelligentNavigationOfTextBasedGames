# Intelligent Navigation of Text-Based Games

## Generating Games
To generate a game, run:
    `tw-make.py twcc_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)`
where:
    - `mode` is easy if there are no distractor nodes and hard if there are a lot.
    - `level` is the number of rooms on the path to the goal.
    - `gamesize` is the number of maps.
    - `step` is ???.
    - `seed` is the random seed.
    - `split` determines which group of data these maps belong to.
Then edit the `game_folder` variable in file `test_game.py` to point to the new game folder (which can likely be found in the `gen_games` folder). Note that the `gen_games` folder already contains games of all three modes for levels 1-3, where each game contains 10 maps.

## Selecting an Agent
To select an agent, edit the `agent` variable in file `test_game.py`. Import the agent, if is not already imported.

## Testing the Agent
To test the agent, run `python 3 test_game.py`.
