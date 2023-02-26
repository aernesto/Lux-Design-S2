import json
import logging
from typing import Dict
import sys
from argparse import Namespace

from agents_experiments import ControlledAgent
from lux.config import EnvConfig
from lux.kit import process_obs, process_action
logging.basicConfig(
filename='/home/adrian_admin/prog/kaggle/lux-ai-2/Lux-Design-S2/data/experiments/adrian/main_v2_37.log',
level=logging.WARNING
)
agent_dict = dict(
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = ControlledAgent(
        player,
        env_cfg,
        radius=130,
        threshold=15
        )
        agent_prev_obs[player] = dict()
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step,
                      json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if obs["real_env_steps"] < 0:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    #  step = 0
    #  player_id = 0
    #  configurations = None
    i = 0
    while True:
        inputs = read_input()  # TODO: find out what inputs looks like
        obs = json.loads(inputs)

        observation = Namespace(
            **dict(step=obs["step"],
                   obs=json.dumps(obs["obs"]),
                   remainingOverageTime=obs["remainingOverageTime"],
                   player=obs["player"],
                   info=obs["info"]))
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        logging.debug('{i} {op}'.format(i=i, op=obs["player"]))
        logging.debug('{i} inputs=<START>{inputs}<STOP>'.format(i=i,
                                                                inputs=inputs))
        logging.debug('{i} actions=<START>{actions}<STOP>\n'.format(
            i=i, actions=actions))
        print(json.dumps(actions))
