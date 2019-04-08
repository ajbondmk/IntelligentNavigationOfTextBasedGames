"""
The CLI for running experiments to train and/or test agents on TextWorld Coin Collector games.
"""


import click
from run_agents import random_agent_eval
from run_agents import dqn_agent_eval_single
from run_agents import dqn_agent_eval_multiple
from run_agents import dqn_agent_eval_zero_shot


try:

    @click.group()
    def top_level_group():
        pass

    @top_level_group.command()
    @click.argument('world_folder', type=click.Path(exists=True))
    def random(world_folder):
        random_agent_eval(world_folder)

    @top_level_group.group()
    def dqn():
        pass

    @dqn.command()
    @click.argument('world_folder', type=click.Path(exists=True))
    def single(world_folder):
        dqn_agent_eval_single(world_folder)

    @dqn.command()
    @click.argument('world_folder', type=click.Path(exists=True))
    def multiple(world_folder):
        dqn_agent_eval_multiple(world_folder)

    @dqn.command()
    @click.argument('world_folder', type=click.Path(exists=True))
    @click.argument('test_world_folder', type=click.Path(exists=True))
    def zero_shot(world_folder, test_world_folder):
        dqn_agent_eval_zero_shot(world_folder, test_world_folder)

    if __name__ == "__main__":
        top_level_group()

except KeyboardInterrupt:
    print(" KeyboardInterrupt\n")
