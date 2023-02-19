# -*- coding: utf-8 -*-
"""code relative to observations"""
import numpy as np
from typing import Dict
import logging

PLAYER_TYPE = 'player'
ROBOT_TYPES = {'light': 'light robot', 'heavy': 'heavy robot'}
FACTORY_TYPE = 'factory'


def flip_name(n):
    return 'player_0' if n == 'player_1' else 'player_1'


class CenteredObservation:
    def __init__(self, obs_dict: Dict, player_name: str):
        self.dict_obj = obs_dict
        assert player_name.startswith('player')
        self._my_player_name = player_name
        self._opp_name = flip_name(self.myself)

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
    def opp_factories(self):
        return self.dict_obj['factories'][self.opponent]

    @property
    def ice_map(self):
        return self.dict_obj['board']['ice']

    @property
    def ore_map(self):
        return self.dict_obj['board']['ore']

    @property
    def my_units(self):
        return self.dict_obj['units'][self.my_player_name]

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


class RobotCenteredObservation(CenteredObservation):
    def __init__(self, obs_dict: Dict, unit_id: str):
        self.dict_obj = obs_dict
        assert unit_id.startswith('unit')
        self._my_unit_id = unit_id
        self._my_player_name = self._infer_player_from_unit_id(unit_id)
        self._opp_name = flip_name(self.my_player_name)

    def _infer_player_from_unit_id(self, uid: str):
        try:
            for player_name, unit_dict in self.dict_obj['units'].items():
                if uid in unit_dict.keys():
                    return player_name
            raise ValueError(f"unit with ID {uid} not found in obs_dict")
        except TypeError:
            logging.debug(f'{type(self)=}    {dir(self)=}')
            raise

    def _get_type(self):
        return ROBOT_TYPES[self.state['unit_type']]

    @property
    def my_type(self):
        return self._get_type()

    @property
    def myself(self):
        return self._my_unit_id

    @property
    def state(self):
        return self.dict_obj['units'][self.my_player_name][self.myself]

    @property
    def position(self):
        return self.state['pos']

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

    @property
    def queue(self):
        return self.state['action_queue']


class FactoryCenteredObservation(CenteredObservation):
    def __init__(self, obs_dict: Dict, unit_id: str):
        self.dict_obj = obs_dict
        assert unit_id.startswith('factory')
        self._my_unit_id = unit_id
        self._my_player_name = self._infer_player_from_unit_id(unit_id)
        self._opp_name = flip_name(self.my_player_name)

    def _infer_player_from_unit_id(self, uid: str):
        try:
            for player_name, unit_dict in self.dict_obj['factories'].items():
                if uid in unit_dict.keys():
                    return player_name
            raise ValueError(f"unit with ID {uid} not found in obs_dict")
        except TypeError:
            logging.debug(f'{type(self)=}    {dir(self)=}')
            raise

    @property
    def my_type(self):
        return FACTORY_TYPE

    @property
    def myself(self):
        return self._my_unit_id

    @property
    def state(self):
        return self.dict_obj['factories'][self.my_player_name][self.myself]

    @property
    def position(self):
        return self.state['pos']

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
