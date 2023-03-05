# -*- coding: utf-8 -*-
import logging
from obs import CenteredObservation, RobotCenteredObservation, FactoryCenteredObservation, PlantId
from luxai_s2.env import EnvConfig
import numpy as np
from codetiming import Timer
from functools import reduce
import networkx as nx
from typing import Sequence, List, Tuple, Optional
from space import CartesianPoint, xy_iter
Array = np.ndarray


def invert_dict(d):
    return {v: k for k, v in d.items()}


_MOVE = 'move'
"""String representing movement in this module."""

_TRANSFER = 'transfer _3 amount of _RESOURCE'
"""String representing transfer in this module."""

_PICKUP = 'pickup _3 amount of _RESOURCE'
"""String representing pickup in this module."""

_DIG = 'dig'
"""String representing dig in this module."""

_SELF_DESTRUCT = 'self-destruct'
"""String representing self-destruct in this module."""

_RECHARGE = 'recharge X'
"""String representing recharge in this module."""

_0 = {
    0: _MOVE,
    1: _TRANSFER,
    2: _PICKUP,
    3: _DIG,
    4: _SELF_DESTRUCT,
    5: _RECHARGE
}
"""_0 is a dict mapping integer index to action string."""

_TYPE = invert_dict(_0)
"""_TYPE is a dict mapping action string to corresponding int."""


def _act_str2int(s: str): return _TYPE[s]


def _act_int2str(i: int): return _0[i]


_CENTER, _UP, _RIGHT, _DOWN, _LEFT = 'center', 'up', 'right', 'down', 'left'

_1 = {i: s for i, s in enumerate([_CENTER, _UP, _RIGHT, _DOWN, _LEFT])}
"""_1 is a dict mapping integers to direction strings"""

_DIRECTION = invert_dict(_1)
"""_DIRECTION is a dict mapping direction strings to integers"""

_MIRROR_DIRECTIONS = {
    'center': 'center',
    'left': 'right',
    'right': 'left',
    'up': 'down',
    'down': 'up'
}


def _dir_str2int(s: str): return _DIRECTION[s]
def _dir_int2str(i: int): return _1[i]


_DEFAULT_DIRECTION: int = _DIRECTION[_CENTER]

_ICE, _ORE, _WATER, _METAL, _POWER = 'ice', 'ore', 'water', 'metal', 'power'

_2 = {i: s for i, s in enumerate([_ICE, _ORE, _WATER, _METAL, _POWER])}
"""_2 is a dict mapping integers to resource strings"""

_RESOURCE = invert_dict(_2)
"""_RESOURCE is a dict mapping resource strings to integers"""

_DEFAULT_RESOURCE = 0
_DEFAULT_AMOUNT = 1

_REPEAT = 4
_N = 5

"""
_3 = amount
"""


def _transfer(amount: int, resource: str, direction: str = _CENTER, repeat: int = 0, n: int = 1):
    return np.array([
        _act_str2int(_TRANSFER), 
        _dir_str2int(direction), 
        _RESOURCE[resource], 
        amount, 
        repeat, 
        n
    ])


def _pickup(amount: int, resource: str, repeat: int = 0, n: int = 1):
    return np.array([
        _act_str2int(_PICKUP), _DEFAULT_DIRECTION, _RESOURCE[resource], amount,
        repeat, n
    ])


def _move(direction: str, repeat: int = 0, n: int = 1):
    return np.array([
        _act_str2int(_MOVE), _dir_str2int(direction), _DEFAULT_RESOURCE,
        _DEFAULT_AMOUNT, repeat, n
    ])


def flip_movement_queue(mv_queue: Sequence[Array]):
    new_queue = []
    for move in reversed(mv_queue):
        if move[0] != 0:
            raise ValueError(f"{move} is not a movement array")
        new_move = move.copy()
        old_move_direction_int = move[1]
        old_move_direction_str = _dir_int2str(old_move_direction_int)
        new_move_direction_str = _MIRROR_DIRECTIONS[old_move_direction_str]
        new_move[1] = _dir_str2int(new_move_direction_str)
        new_queue.append(new_move)
    return new_queue


def _dig(resource: str, repeat: int = 0, n: int = 1):
    return np.array([
        _act_str2int(
            _DIG), _DEFAULT_DIRECTION, _RESOURCE[resource], _DEFAULT_AMOUNT,
        repeat, n
    ])


def compress_queue(q: Sequence[Array]):
    def combine(old_list: Sequence, new_item: Array):
        if len(old_list) == 0:
            return [new_item]

        last = old_list[-1]
        if (new_item[:-1] == last[:-1]).all():
            last[-1] += new_item[-1]
        else:
            old_list.append(new_item)
        return old_list

    return list(reduce(combine, q, []))


def format_repeat(
        seq: Sequence[Array], 
        rep_val: Optional[int], 
        omit_ix: Optional[Sequence[int]] = None
        ):
    """Populate the repeat index of all actions in queue.

    Args:
        seq (Sequence[Array]): queue to modify (after copying)
        rep_val (Optional[int]): If a fixed value, will assign it to all actions 
               in queue. If None, will set the repeat value to the n-value.
        omit_ix (Optional[Sequence[int]]): indices in queue for which to leave repeat 
               unchanged. 
    Returns:
        List[Array]: queue
    """
    new_seq = [arr.copy() for arr in seq]
    if omit_ix is None:
        omit_ix = {}
    for ix, arr in enumerate(new_seq):
        if ix in omit_ix:
            continue
        if rep_val is None:
            arr[_REPEAT] = arr[_N]
        else:
            arr[_REPEAT] = rep_val
    return new_seq


class MapPlanner:
    """Class that helps planning robot moves."""
    class_id = 0

    def __init__(self, obs: CenteredObservation):
        self.class_id += 1
        self.obs = obs
        self.network = self._build_network()
        self.planner_timer = Timer(
            f"MapPlanner_timer {self.class_id}",
            text="network operations: {:.2f}",
            logger=None
        )

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
        for point in G.nodes:
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
            enemy_plants_ += [center] + list(center.surrounding_neighbors)
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
        self.myself = self.obs.myself
        self.pos = self.obs.pos
        logging.debug('RobotEnacter.cost_type={}'.format(self.cost_type))

    def dig_cycle(
            self,
            target_loc: CartesianPoint,
            resource: str,
            cycle_start_pos: CartesianPoint,
            dig_n: int = 5,
    ):
        logging.debug('--debug from RobotEnacter.ice_cycle')

        # TODO: go to start
        start = self.planner._nx_shortest_path(
            self.obs.pos, cycle_start_pos, cost_type=self.cost_type)
        start = self.planner.nx_path_to_action_sequence(start)
        start = compress_queue(start)

        # get shortest path from robot to ice
        go_nx_path = self.planner._nx_shortest_path(self.obs.pos,
                                                    target_loc,
                                                    cost_type=self.cost_type)
        logging.debug('robot pos={} target_loc={}'.format(
            self.obs.pos, target_loc))

        # translate path to action queue
        go_nx_path = self.planner.nx_path_to_action_sequence(go_nx_path)
        go_nx_path = compress_queue(go_nx_path)
        go_nx_path = format_repeat(go_nx_path, rep_val=None)
        logging.debug('go path={}'.format(go_nx_path))

        # append dig action
        # TODO: factor in power cost in below logic
        dig_queue = [_dig(resource, dig_n, dig_n)]
        logging.debug('go + dig={}'.format(dig_queue))

        # append return path
        # TODO: check correctness of next two lines
        # return_nx_path = self.planner._nx_shortest_path(target_loc, self.obs.pos, cost_type=self.cost_type)
        return_nx_path = flip_movement_queue(go_nx_path)

        # append transfer resource action
        transfer_queue = [_transfer(self.obs.state['cargo'][resource], resource, repeat=1, n=1)]

        # append pickup action
        # TODO: only pickup what is necessary for next cycle
        pickup_queue = [_pickup(500, _POWER, repeat=1)]

        queue = start + go_nx_path + dig_queue
        queue += return_nx_path + transfer_queue + pickup_queue
        logging.debug('start + go + dig + return + pickup={}'.format(queue))

        # queue = compress_queue(queue)

        logging.debug('--END debug from RobotEnacter.ice_cycle')

        # TODO: avoid hard-coding queue length below
        logging.info("""
sending robot {} currently at position {}
on an {} cycle with start tile {} and target tile {}
Full compressed queue is {}
""".format(self.myself, self.pos, resource, cycle_start_pos, target_loc, queue))
        # breakpoint()
        return queue[:20]


if __name__ == "__main__":
    pass
