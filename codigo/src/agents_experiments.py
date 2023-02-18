# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass, field
from luxai_s2.team import FactionTypes, Team
from luxai_s2.factory import Factory
from luxai_s2.state import State
from luxai_s2.env import EnvConfig, Board
from typing import Dict, Optional
from luxai_s2.unit import Unit
from collections import OrderedDict
import numpy as np
from luxai_s2.utils import animate, my_turn_to_place_factory


def reset_w_custom_board(environment, seed, custom_board):
    environment.agents = environment.possible_agents[:]
    environment.env_steps = 0
    if seed is not None:
        environment.seed_val = seed
        environment.seed_rng = np.random.RandomState(seed=seed)
    else:
        environment.seed_val = np.random.randint(0, 2**32 - 1)
        environment.seed_rng = np.random.RandomState(seed=environment.seed_val)

    environment.state: State = State(
        seed_rng=environment.seed_rng,
        seed=environment.seed_val,
        env_cfg=environment.state.env_cfg,
        env_steps=0,
        board=custom_board,
    )

    environment.max_episode_length = environment.env_cfg.max_episode_length
    for agent in environment.possible_agents:
        environment.state.units[agent] = OrderedDict()
        environment.state.factories[agent] = OrderedDict()
        if environment.collect_stats:
            environment.state.stats[agent] = create_empty_stats()
    obs = environment.state.get_obs()
    observations = {agent: obs for agent in environment.agents}
    return observations, environment


class IdleAgent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.debug_act = self.act

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        actions = dict()
        try:
            factories_to_place = obs['teams'][
                self.player]['factories_to_place']
        except AttributeError:
            print(obs['teams'][self.player])
            raise
        my_turn_to_place = my_turn_to_place_factory(
            obs['teams'][self.player]['place_first'], step)

        if factories_to_place > 0 and my_turn_to_place:
            potential_spawns = np.array(
                list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
            #             logging.info(f'IdleAgent {potential_spawns.shape=}')
            selected_location_ix = -1
            spawn_loc = potential_spawns[selected_location_ix]
            #             logging.info(f'IdleAgent {spawn_loc=}')
            return dict(spawn=spawn_loc, metal=150, water=150)
        return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        return actions


def interact(env,
             agents,
             steps,
             animate_: str = '',
             break_at_first_action: bool = False,
             debug: bool = False,
             custom_board: Optional[Board] = None):
    # reset our env
    if custom_board is None:
        raise NotImplementedError()
    else:
        obs, env = reset_w_custom_board(env,
                                        seed=41,
                                        custom_board=custom_board)
    np.random.seed(0)
    imgs = []
    step = 0

    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        if debug:
            logging.debug(f'{step=}')
    done = False

    inner_counter = 0
    while not done:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            if debug:
                a = agents[player].debug_act(step, o)
                logging.debug(f"{step=}  {inner_counter=}")
            else:
                a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]
        inner_counter += 1
        if break_at_first_action and inner_counter == 2:
            break
    if animate_:
        logging.info(f'writing {animate_}')
        return animate(imgs, filename=animate_)
    else:
        return obs


if __name__ == "__main__":
    pass
