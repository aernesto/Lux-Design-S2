# -*- coding: utf-8 -*-
"""code relative to observations"""
import numpy as np
from typing import Dict
import logging
from luxai_s2.config import EnvConfig
from space import CartesianPoint
from collections import namedtuple

RobotId = namedtuple("RobotId", "unit_id type", module=__name__)
PlantId = namedtuple("PlantId", "unit_id pos", module=__name__)
PlantAssignment = namedtuple("PlantAssignment", "unit_id pos tile", module=__name__)

# TODO: Not sure whether below line is good practice
ENV_CONFIG = EnvConfig()
PLAYER_TYPE = 'player'
FACTORY_TYPE = 'factory'
HEAVY_TYPE = 'HEAVY'
LIGHT_TYPE = 'LIGHT'


def flip_name(n):
    return 'player_0' if n == 'player_1' else 'player_1'


class CenteredObservation:
    def __init__(self, obs_dict: Dict, player_name: str):
        self.dict_obj = obs_dict
        assert player_name.startswith('player')
        self._my_player_name = player_name
        self._opp_name = flip_name(self.myself)
        self.ice_pos = set([
            CartesianPoint(x, y, self.board_length)
            for x, y in zip(*self.ice_map.nonzero())
        ])
        self.ore_pos = set([
            CartesianPoint(x, y, self.board_length)
            for x, y in zip(*self.ore_map.nonzero())
        ])
        self.resource_pos = self.ice_pos.union(self.ore_pos)

    @property
    def board(self):
        try:
            return self.dict_obj['board']
        except KeyError:
            logging.debug('available keys={}'.format(self.dict_obj.keys()))
            raise

    @property
    def my_type(self):
        return PLAYER_TYPE

    @property
    def myself(self):
        return self._my_player_name

    @property
    def my_player_name(self):
        return self._my_player_name

    @property
    def opponent(self):
        return self._opp_name

    @property
    def my_factories(self):
        return self.dict_obj['factories'][self.my_player_name]
    
    @property
    def num_factories(self):
        return len(self.my_factories)

    @property
    def opp_factories(self):
        return self.dict_obj['factories'][self.opponent]

    @property
    def ice_map(self):
        return self.board['ice']

    @property
    def board_length(self):
        return len(self.ice_map)

    @property
    def rubble_map(self):
        return self.board['rubble']

    @property
    def ore_map(self):
        return self.board['ore']

    @property
    def my_units(self):
        return self.dict_obj['units'][self.my_player_name]

    @property
    def robot_ids(self):
        return [RobotId(k, v['unit_type']) for k, v in self.my_units.items()]

    @property
    def factory_ids(self):
        return [PlantId(k, CartesianPoint(*v['pos'], self.board_length)) for k, v in self.my_factories.items()]

    @property
    def my_heavy_units(self):
        return {k: v for k, v in self.my_units.items() if v.unit_type == HEAVY_TYPE}

    @property
    def my_light_units(self):
        return {k: v for k, v in self.my_units.items() if v.unit_type == LIGHT_TYPE}

    @property
    def my_team(self):
        return self.dict_obj['teams'][self.my_player_name]

    @property
    def opp_units(self):
        return self.dict_obj['units'][self.opponent]

    @property
    def factories_ranked_by_power(self):
        dtype = [('power', float), ('factory_id', '<U10')]
        all_factories = []
        for factory_id, fac_state in self.my_factories.items():
            all_factories.append((fac_state['power'], factory_id))
        a = np.array(all_factories, dtype=dtype)  # create a structured array
        return np.flip(np.sort(a, order='power'))

    @property
    def total_factories_cargo(self):
        cargos = [f['cargo'] for f in self.my_factories.values()]
        if cargos:
            keys = cargos[0].keys()
            return {k: sum(d[k] for d in cargos) for k in keys}
        return {'ice': 0, 'metal': 0, 'ore': 0, 'water': 0}

    @property
    def total_robots_cargo(self):
        cargos = [f['cargo'] for f in self.my_units.values()]
        if cargos:
            keys = cargos[0].keys()
            return {k: sum(d[k] for d in cargos) for k in keys}
        return {'ice': 0, 'metal': 0, 'ore': 0, 'water': 0}

    def generate_factory_obs(self):
        for fac_id in self.my_factories.keys():
            yield FactoryCenteredObservation(self.dict_obj, fac_id)
    
    def generate_robot_obs(self):
        for robot_id in self.my_units.keys():
            yield RobotCenteredObservation(self.dict_obj, robot_id)


class RobotCenteredObservation(CenteredObservation):
    def __init__(self, obs_dict: Dict, unit_id: str):
        self.dict_obj = obs_dict
        assert unit_id.startswith('unit')
        self.unit_id = unit_id
        self._my_player_name = self._infer_player_from_unit_id(unit_id)
        self._opp_name = flip_name(self.my_player_name)

    def _infer_player_from_unit_id(self, uid: str):
        try:
            for player_name, unit_dict in self.dict_obj['units'].items():
                if uid in unit_dict.keys():
                    return player_name
            raise ValueError(
                "unit with ID {uid} not found in obs_dict".format(uid=uid))
        except TypeError:
            logging.debug('{s}  {d}'.format(s=type(self), d=dir(self)))
            raise

    def _get_type(self):
        return self.state['unit_type']

    @property
    def my_type(self):
        return self._get_type()
    
    @property
    def cost_type(self):  # used for weight choosing in networkx
        return 'heavy_weight' if self.my_type == 'HEAVY' else 'light_weight'

    @property
    def myself(self):
        return RobotId(self.unit_id, self.my_type)

    @property
    def state(self):
        return self.dict_obj['units'][self.my_player_name][self.unit_id]

    @property
    def pos(self):
        return CartesianPoint(*self.state['pos'], self.board_length)

    @property
    def power(self):
        return self.state['power']

    @property
    def ice(self):
        return self.state['cargo']['ice']

    @property
    def ice_capacity(self):
        return ENV_CONFIG.ROBOTS[self.my_type].CARGO_SPACE - self.ice

    @property
    def ore(self):
        return self.state['cargo']['ore']

    @property
    def water(self):
        return self.state['cargo']['water']

    @property
    def metal(self):
        return self.state['cargo']['metal']

    @property
    def queue(self):
        return self.state['action_queue']

    @property
    def queue_is_empty(self):
        return len(self.queue) == 0

    @property
    def dig_yield(self):
        return ENV_CONFIG.ROBOTS[self.my_type].DIG_RESOURCE_GAIN


class FactoryCenteredObservation(CenteredObservation):
    def __init__(self, obs_dict: Dict, unit_id: str):
        assert isinstance(obs_dict, dict), "obs_dict is type={}".format(
            type(obs_dict))
        self.dict_obj = obs_dict
        assert unit_id.startswith('factory')
        self.unit_id = unit_id
        self._my_player_name = self._infer_player_from_unit_id(unit_id)
        self._opp_name = flip_name(self.my_player_name)

    def _infer_player_from_unit_id(self, uid: str):
        try:
            for player_name, unit_dict in self.dict_obj['factories'].items():
                if uid in unit_dict.keys():
                    return player_name
            raise ValueError(
                "unit with ID {uid} not found in obs_dict".format(uid=uid))
        except TypeError:
            logging.error('{}'.format(self.dict_obj))
            raise

    @property
    def my_type(self):
        return FACTORY_TYPE

    @property
    def myself(self):
        return PlantId(self.unit_id, self.pos)

    @property
    def state(self):
        return self.dict_obj['factories'][self.my_player_name][self.unit_id]

    @property
    def pos(self):
        return CartesianPoint(*self.state['pos'], self.board_length)

    @property
    def power(self):
        return self.state['power']

    @property
    def ice(self):
        return self.state['cargo']['ice']

    @property
    def ore(self):
        return self.state['cargo']['ore']

    @property
    def water(self):
        return self.state['cargo']['water']

    @property
    def metal(self):
        return self.state['cargo']['metal']


if __name__ == "__main__":
    pass
