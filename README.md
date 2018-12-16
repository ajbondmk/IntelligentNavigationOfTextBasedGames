# Intelligent Navigation of Text-Based Games

## Generating Games
To generate a game, run:
    `tw_make.py twcc_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)`
where:
    - `mode` is easy if there are no distractor nodes and hard if there are a lot.
    - `level` is the number of rooms on the path to the goal.
    - `gamesize` is the number of maps.
    - `step` is ???.
    - `seed` is the random seed.
    - `split` determines which group of data these maps belong to.
Then edit the `game_folder` variable in file `test_game.py` to point to the new game folder.

## Selecting an Agent
To select an agent, edit the `agent` variable in file `test_game.py`. Import the agent, if is not already imported.

## Testing the Agent
To test the agent, run `python 3 test_game.py`.
