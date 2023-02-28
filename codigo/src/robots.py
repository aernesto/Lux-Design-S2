# -*- coding: utf-8 -*-
import logging
from obs import CenteredObservation, RobotCenteredObservation, FactoryCenteredObservation, PlantId
from luxai_s2.env import EnvConfig
import numpy as np
from codetiming import Timer
import networkx as nx
from typing import Sequence, List, Tuple
from space import CartesianPoint, xy_iter
Array = np.ndarray


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


def _move(direction: str, repeat: int = 0, n: int = 1):
    return np.array([
        _TYPE[_MOVE], _DIRECTION[direction], _DEFAULT_RESOURCE,
        _DEFAULT_AMOUNT, repeat, n
    ])


def flip_movement_queue(mv_queue: Sequence[Array]):
    new_queue = []
    for move in reversed(mv_queue):
        new_queue.append(_move(_1[move[0]], *move[-2:]))
    return new_queue


def _dig(resource, repeat: int = 0, n: int = 1):
    return np.array([
        _TYPE[_DIG], _DEFAULT_DIRECTION, _RESOURCE[resource], _DEFAULT_AMOUNT,
        repeat, n
    ])


class MapPlanner:
    """Class that helps planning robot moves."""
    class_id = 0

    def __init__(self, obs: CenteredObservation):
        self.class_id += 1
        self.obs = obs
        self.network = self._build_network()
        self.planner_timer = Timer(
            f"MapPlanner_timer {self.class_id}", text="network operations: {:.2f}", logger=logging.info)

    @property
    def board_length(self):
        return len(self.obs.ice_map)

    @property
    def rubble(self):
        return self.obs.rubble_map

    @property
    def ice(self):
        return self.obs.ice_map

    @property
    def ore(self):
        return self.obs.ore_map

    def _build_network(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(xy_iter(self.board_length))
        edges = []
        for point in xy_iter(self.board_length):
            rb = self.rubble[point.x, point.y]
            factories = self.obs.my_factories
            for fac_id in factories:
                fac_pos = FactoryCenteredObservation(
                    self.obs.dict_obj, fac_id).pos
                fac_tiles = {fac_pos}.union(fac_pos.surrounding_neighbors)
                if point in fac_tiles:
                    rb = 0
                    break
            # TODO: fetch the library's real cost functions
            light_weight = np.floor(1 + 0.05 * rb)
            heavy_weight = np.floor(20 + rb)
            wd = dict(light_weight=light_weight, heavy_weight=heavy_weight)
            if not point.at_top_edge:
                edges.append((point.top_neighbor, point, wd.copy()))

            if not point.at_bottom_edge:
                edges.append((point.bottom_neighbor, point, wd.copy()))

            if not point.at_left_edge:
                edges.append((point.left_neighbor, point, wd.copy()))

            if not point.at_right_edge:
                edges.append((point.right_neighbor, point, wd.copy()))
        G.add_edges_from(edges)

        enemy_plants_ = []
        for fd in self.obs.opp_factories.values():
            center = CartesianPoint(*fd['pos'])
            enemy_plants_ += [
                center, center.left_neighbor,
                center.left_neighbor.top_neighbor, center.top_neighbor,
                center.top_neighbor.right_neighbor, center.right_neighbor,
                center.right_neighbor.bottom_neighbor, center.bottom_neighbor,
                center.bottom_neighbor.left_neighbor
            ]
        G.remove_nodes_from(enemy_plants_)
        return G

    def _nx_shortest_path(self, node1, node2, cost_type: str):
        with self.planner_timer:
            return nx.shortest_path(self.network,
                                    source=node1,
                                    target=node2,
                                    weight=cost_type)

    def _action_translator(self, dx, dy):
        if dx == 1 and dy == 0:
            return _move(_RIGHT)
        if dx == -1 and dy == 0:
            return _move(_LEFT)
        if dx == 0 and dy == 1:
            return _move(_DOWN)
        if dx == 0 and dy == -1:
            return _move(_UP)

    def heavy_distance(self, p1: CartesianPoint, p2: CartesianPoint) -> float:
        """Distance for a heavy robot (in units of power) between 2 points"""
        path = self._nx_shortest_path(p1, p2, cost_type='heavy_weight')
        return nx.path_weight(self.network, path, 'heavy_weight')

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

    def resources_radial_count(
            self,
            center: CartesianPoint,
            radius: float
    ) -> Tuple[int, List[CartesianPoint], int, List[CartesianPoint]]:
        """Count resources in radius"""
        # get all points within radius
        new_graph = nx.generators.ego_graph(self.network,
                                            center,
                                            radius=radius,
                                            distance='heavy_weight')

        ice_count, ore_count = 0, 0
        ice_set, ore_set = [], []
        # loop over them and count
        for node in new_graph.nodes:
            if self.ice[node.x, node.y]:
                ice_count += 1
                ice_set.append(node)
            if self.ore[node.x, node.y]:
                ore_count += 1
                ore_set.append(node)

        return ice_count, ice_set, ore_count, ore_set

    def rank_factories_by_distance_from(self, ref: CartesianPoint, cost_type: str) -> Array:
        """Ranks factories by (cost_type)-distance to reference location.

        Args:
            ref (CartesianPoint): _description_
            cost_type (str): _description_

        Returns:
            Array: structured array of plants
        """
        # TODO: see if a cache of distances can be built lazily
        factory_list = []  # will contain tuples to build structured numpy array
        for fac in self.obs.factory_ids:
            best_distance = np.inf
            # we need the best factory tile (which cannot be the center)
            for tile in fac.pos.surrounding_neighbors:
                dist_to_tile = nx.path_weight(
                    self.network,
                    nx.shortest_path(self.network, ref,
                                     tile, weight=cost_type),
                    weight=cost_type
                )
                if dist_to_tile < best_distance:
                    best_distance = dist_to_tile
                    best_tile = tile
            factory_list.append((fac, best_tile, best_distance))
        dtype = [('fac', PlantId), ('tile', CartesianPoint), ('dist', float)]
        a = np.array(factory_list, dtype=dtype)  # create a structured array
        return np.sort(a, order='dist')


class RobotEnacter:
    def __init__(self, robot_obs: RobotCenteredObservation,
                 env_cfg: EnvConfig):
        self.obs = robot_obs
        self.conf = env_cfg
        self.planner = MapPlanner(self.obs)
        self.cost_type = self.obs.my_type.lower() + '_weight'
        logging.debug('RobotEnacter.cost_type={}'.format(self.cost_type))

    def compress_queue(self, q, repeat: bool = False):
        new_queue = []

        original = q[0].copy()
        new_action = original.copy()
        for i, action in enumerate(q[1:]):
            if (original == action).all():
                new_action[5] += 1
            else:
                new_action[4] = repeat
                new_queue.append(new_action)
                new_action = action.copy()
                original = action.copy()
                if i == len(q) - 2:
                    new_action[4] = repeat
                    new_queue.append(new_action)
        return new_queue

    def dig_cycle(
            self,
            ice_loc: CartesianPoint,
            cycle_start_pos: CartesianPoint,
            repeat: bool = True,
            dig_n: int = 5
    ):
        logging.debug('--debug from RobotEnacter.ice_cycle')

        # TODO: go to start
        start = self.planner._nx_shortest_path(
            self.obs.pos, cycle_start_pos, cost_type=self.cost_type)
        start = self.planner.nx_path_to_action_sequence(start)
        start = self.compress_queue(start, repeat=False)

        # get shortest path from robot to ice
        go_nx_path = self.planner._nx_shortest_path(self.obs.pos,
                                                    ice_loc,
                                                    cost_type=self.cost_type)
        logging.debug('robot pos={} ice_loc={}'.format(self.obs.pos, ice_loc))

        # translate path to action queue
        go_nx_path = self.planner.nx_path_to_action_sequence(go_nx_path)
        go_nx_path = self.compress_queue(go_nx_path, repeat=True)
        logging.debug('go path={}'.format(queue))

        # append dig action
        # TODO: factor in power cost in below logic
        _repeat = True
        dig_queue = [_dig(_ICE, _repeat, dig_n)]
        logging.debug('go + dig={}'.format(queue))

        # append return path
        # TODO: check correctness of next two lines
        # return_nx_path = self.planner._nx_shortest_path(ice_loc, self.obs.pos, cost_type=self.cost_type)
        return_nx_path = flip_movement_queue(go_nx_path)

        queue = start + go_nx_path + dig_queue + return_nx_path
        logging.debug('go + pickup + return={}'.format(queue))

        queue = self.compress_queue(queue)

        logging.debug('--END debug from RobotEnacter.ice_cycle')

        # TODO: avoid hard-coding queue length below
        return queue[:20]


if __name__ == "__main__":
    pass
