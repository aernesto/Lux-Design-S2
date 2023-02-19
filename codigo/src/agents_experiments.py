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
from obs import (CenteredObservation, RobotCenteredObservation,
                 FactoryCenteredObservation)
from robots import Enacter
from plants import PlantEnacter


class ControlledAgent:
    stats = []

    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg

    @property
    def heavy_price(self):
        return {
            'metal': self.env_cfg.ROBOTS['HEAVY'].METAL_COST,
            'power': self.env_cfg.ROBOTS['HEAVY'].POWER_COST
        }

    @property
    def light_price(self):
        return {
            'metal': self.env_cfg.ROBOTS['LIGHT'].METAL_COST,
            'power': self.env_cfg.ROBOTS['LIGHT'].POWER_COST
        }

    def monitor(self, step, obs):
        o = CenteredObservation(obs, self.player)
        assert step == len(self.stats) + 1, f"{step=}  {len(self.stats)=}"
        c = o.total_factories_cargo
        c.update({
            'number': len(o.my_factories),
            'power': np.sum(o.factories_ranked_by_power['power']),
        })
        self.stats.append({'factories_total': c})

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            self.monitor(step, obs)
            myteam = obs['teams'][self.player]
            water_left = myteam['water']
            metal_left = myteam['metal']

            # how many factories you have left to place
            factories_to_place = myteam['factories_to_place']

            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                myteam['place_first'], step)

            if factories_to_place > 0 and my_turn_to_place:
                inner_list = list(
                    zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
                potential_spawns = np.array(inner_list)
                selected_location_ix = 0
                spawn_loc = potential_spawns[selected_location_ix]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.monitor(step, obs)
        observation = CenteredObservation(obs, self.player)
        actions = dict()

        total_heavy_quota = observation.ice_map.sum(
        ) + observation.ore_map.sum()
        existing_heavies = len(observation.my_units)
        num_heavy_to_build = total_heavy_quota - existing_heavies
        for fac in observation.factories_ranked_by_power:
            factory_id = fac['factory_id']
            try:
                plant_obs = FactoryCenteredObservation(obs, factory_id)
            except AttributeError:
                logging.debug(f"{fac=}")
                raise
            plant_enacter = PlantEnacter(plant_obs, self.env_cfg)
            if num_heavy_to_build:
                enough_metal = plant_obs.metal >= self.heavy_price['metal']
                enough_power = plant_obs.power >= self.heavy_price['power']

                if enough_metal and enough_power:
                    actions[factory_id] = plant_enacter.build_heavy()
                    num_heavy_to_build -= 1
            else:
                # TODO: estimate cost of watering
                actions[factory_id] = plant_enacter.water()

        return actions

    def debug_act(self, step: int, obs, remainingOverageTime: int = 60):
        observation = CenteredObservation(obs, self.player)

        # who am I?
        logger.debug(f'I am {observation.myself}')

        # what turn is it?
        logger.debug(f'{step=}  {remainingOverageTime=}')

        self.calls_to_act += 1
        actions = dict()

        # how many factories does each player have, and where?
        def count_and_locate_factories(fac):
            count = len(fac)
            locations = list(v['pos'] for v in fac.values())
            return count, locations

        myteam_ = observation.my_factories
        opp_team_ = observation.opp_factories
        try:
            my_factories = count_and_locate_factories(myteam_)
            logger.debug(f'{my_factories=}')
        except KeyError:
            logger.debug('no factories found in controlled agent\'s team')

        try:
            opponents_factories = count_and_locate_factories(opp_team_)
            logger.debug(f'{opponents_factories=}')
        except KeyError:
            logger.debug('no factories found in opponent agent\'s team')

        # where are ice and ore?
        logger.debug('where are ice and ore?')
        logger.debug(
            f'{observation.ice_map.shape=}  {observation.ice_map.dtype=}  {observation.ice_map.sum()=}'
        )
        logger.debug(
            f'{observation.ore_map.shape=}  {observation.ore_map.dtype=}  {observation.ore_map.sum()=}'
        )

        factories = obs['factories'][self.player]
        self.factories = factories

        for unit_id, factory in factories.items():
            if factory['power'] >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                    factory['cargo']['metal'] >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                try:
                    actions[unit_id] = 1  # build heavy
                except AttributeError:
                    print(obs)
                    raise


#         logging.debug(f'max queue size={self.env_cfg.UNIT_ACTION_QUEUE_SIZE}')

        for unit_id in observation.my_units.keys():
            robot_obs = RobotCenteredObservation(obs, unit_id)
            robot = Enacter(robot_obs, self.env_cfg)

            logging.debug(f'Hi, I am robot {robot_obs.myself}')
            if not robot_obs.queue:
                actions[unit_id] = [robot.move_right(finite=5)]
            logging.debug(f'{robot_obs.state=}')

        self.actions = actions

        # what were my returned actions?
        logger.debug(f'{actions=}')
        return actions


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
