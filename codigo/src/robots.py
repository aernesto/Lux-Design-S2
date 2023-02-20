# -*- coding: utf-8 -*-
import logging
from obs import RobotCenteredObservation
from luxai_s2.env import EnvConfig
import numpy as np
import itertools
import networkx as nx
from space import CartesianPoint


def invert_dict(d):
    return {v: k for k, v in d.items()}


_MOVE = 'move'
_TRANSFER = 'transfer _3 amount of _RESOURCE'
_PICKUP = 'pickup _3 amount of _RESOURCE'
_DIG = 'dig'
_SELF_DESTRUCT = 'self-destruct'
_RECHARGE = 'recharge X'

_0 = {
    0: _MOVE,
    1: _TRANSFER,
    2: _PICKUP,
    3: _DIG,
    4: _SELF_DESTRUCT,
    5: _RECHARGE
}

_TYPE = invert_dict(_0)

_CENTER, _UP, _RIGHT, _DOWN, _LEFT = 'center', 'up', 'right', 'down', 'left'
_1 = {i: s for i, s in enumerate([_CENTER, _UP, _RIGHT, _DOWN, _LEFT])}
_DIRECTION = invert_dict(_1)
_DEFAULT_DIRECTION = _DIRECTION[_UP]

_ICE, _ORE, _WATER, _METAL, _POWER = 'ice', 'ore', 'water', 'metal', 'power'
_2 = {i: s for i, s in enumerate([_ICE, _ORE, _WATER, _METAL, _POWER])}
_RESOURCE = invert_dict(_2)

_DEFAULT_RESOURCE = 0
_DEFAULT_AMOUNT = 1
_DEFAULT_REPEAT = 0
_DEFAULT_N = 1
"""
_3 = amount
_4 = repeat
_5 = n
"""


def _pickup(amount, resource, repeat: int = 0, n: int = 1):
    return np.array([
        _TYPE[_PICKUP], _DEFAULT_DIRECTION, _RESOURCE[resource], amount,
        repeat, n
    ])


def _move(direction, repeat: int = 0, n: int = 1):
    return np.array([
        _TYPE[_MOVE], _DIRECTION[direction], _DEFAULT_RESOURCE,
        _DEFAULT_AMOUNT, repeat, n
    ])


class MapPlanner:
    def __init__(self, board):
        self.board = board
        self.network = self._board_to_network(self.board)

    @property
    def board_length(self):
        return len(self.board['ice'])

    @property
    def rubble(self):
        return self.board['rubble']

    def _board_to_network(self, board):
        def xy_iter():
            """
            returns iterable of CartesianPoint built from board
            """
            return map(
                CartesianPoint._make,
                itertools.product(range(self.board_length),
                                  range(self.board_length)))

        G = nx.MultiDiGraph()
        G.add_nodes_from(xy_iter())
        edges = []
        # x goes left to right
        # y goes top to bottom
        for point in xy_iter():
            weight = self.rubble.T[point.x, point.y]
            if not point.at_top_edge:
                edges.append((point.top_neighbor, point, dict(weight=weight)))

            if not point.get_at_bottom_edge(self.board_length):
                edges.append(
                    (point.bottom_neighbor, point, dict(weight=weight)))

            if not point.at_left_edge:
                edges.append((point.left_neighbor, point, dict(weight=weight)))

            if not point.get_at_right_edge(self.board_length):
                edges.append(
                    (point.right_neighbor, point, dict(weight=weight)))
        G.add_edges_from(edges)
        return G

    def _nx_shortest_path(self, node1, node2):
        return nx.shortest_path(self.network,
                                source=node1,
                                target=node2,
                                weight='weight')

    def _action_translator(self, dx, dy):
        if dx == 1 and dy == 0:
            return _move(_RIGHT)
        if dx == -1 and dy == 0:
            return _move(_LEFT)
        if dx == 0 and dy == 1:
            return _move(_DOWN)
        if dx == 0 and dy == -1:
            return _move(_UP)

    def nx_path_to_action_sequence(self, path):
        try:
            previous_point = path[0]
        except KeyError:
            logging.debug('pb with path')
            logging.debug(path)
            raise
        actions = []
        for point in path[1:]:
            delta_x = point.x - previous_point.x
            delta_y = point.y - previous_point.y
            actions.append(self._action_translator(delta_x, delta_y))
            previous_point = point
        return actions


class RobotEnacter:
    def __init__(self, robot_obs: RobotCenteredObservation,
                 env_cfg: EnvConfig):
        self.obs = robot_obs
        self.conf = env_cfg
        self.planner = MapPlanner(self.obs.dict_obj['board'])

    def compress_queue(self, q):
        new_queue = []

        original = q[0].copy()
        new_action = original.copy()
        for i, action in enumerate(q[1:]):
            try:
                (original == action).all()
            except AttributeError:
                logging.debug('--debug from RobotEnacter.compress_queue')
                logging.debug('original={}'.format(original))
                logging.debug('action={}'.format(action))
                logging.debug('--END debug from RobotEnacter.compress_queue')
                raise
            if (original == action).all():
                new_action[5] += 1
            else:
                new_queue.append(new_action)
                new_action = action.copy()
                original = action.copy()
                if i == len(q) - 2:
                    new_queue.append(new_action)
        return new_queue

    def ice_cycle(self, ice_loc):
        logging.debug('--debug from RobotEnacter.ice_cycle')

        # get shortest path from robot to ice
        go_nx_path = self.planner._nx_shortest_path(self.obs.pos, ice_loc)
        logging.debug('robot pos={} ice_loc={}'.format(self.obs.pos, ice_loc))

        # translate path to action queue
        queue = self.planner.nx_path_to_action_sequence(go_nx_path)
        logging.debug('go path={}'.format(queue))
        # append pickup action
        queue.append(_pickup(self.obs.ice_capacity, _ICE))
        logging.debug('go + pickup={}'.format(queue))

        # append return path
        return_nx_path = [v for v in reversed(go_nx_path)]
        queue += self.planner.nx_path_to_action_sequence(return_nx_path)
        logging.debug('go + pickup + return={}'.format(queue))

        queue = self.compress_queue(queue)

        logging.debug('--END debug from RobotEnacter.ice_cycle')

        # TODO: avoid hard-coding queue length below
        return queue[:20]


if __name__ == "__main__":
    pass
