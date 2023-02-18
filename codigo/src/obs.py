# -*- coding: utf-8 -*-
"""code relative to observations"""
from typing import Dict
import logging


def flip_name(n):
    return 'player_0' if n == 'player_1' else 'player_1'


class CenteredObservation:
    def __init__(self, obs_dict: Dict, player_name: str):
        self.dict_obj = obs_dict
        self._my_player_name = player_name
        self._opp_name = flip_name(self.myself)

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


class RobotCenteredObservation(CenteredObservation):
    def __init__(self, obs_dict: Dict, unit_id: str):
        self.dict_obj = obs_dict
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


if __name__ == "__main__":
    pass
