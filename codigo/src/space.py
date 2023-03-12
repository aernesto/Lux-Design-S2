# -*- coding: utf-8 -*-
import json
import logging
import functools
import itertools
from typing import Iterator
import numpy as np
# from joblib import Memory
# cachedir = 'space_cache'

# memory = Memory(cachedir, verbose=0)
logger = logging.getLogger(__name__)
Array = np.ndarray
try:
    with open('space_lookup.json', 'rt') as fh:
        SPACE_LOOKUP = json.load(fh)
except:
    with open('../space_lookup.json', 'rt') as fh:
        SPACE_LOOKUP = json.load(fh)

INVERSE = {tuple(v['xys']): k for k, v in SPACE_LOOKUP.items()}


def xy_iter(size: int = 48):
    """
    returns iterable of CartesianPoint built from board
    """
    for (x, y), s in zip(itertools.product(range(size), range(size)),
                         [size] * size * size):
        yield CartesianPoint(x, y, s)


def identify_conn_components(r: Array, threshold: int = 0):
    components = set()
    for point in xy_iter(len(r)):
        if r[point.x, point.y] <= threshold:  # 0-rubble tiles
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('visited point={}'.format(point))
                logger.debug('neighbor set={}'.format(point.all_neighbors))
                logger.debug('current components set={}'.format(components))
            touched_components = [
                c for c in components if c.touches_point(point)
            ]
            if len(touched_components):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(touched_components)
                components = components - set(
                    touched_components)  # remove touched
                touched_components += [ConnectedComponent(
                    (point, ))]  # add the point
                merger = ConnectedComponent.union(
                    touched_components)  # merge all
                components.update({merger})  # add them back in
            else:
                components.update({ConnectedComponent([point])})
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('new components set ={}'.format(components))

    return components


# @memory.cache
# class MemoizeMutable:
#     """Memoize(fn) - an instance which acts like fn but memoizes its arguments
#        Will work on functions with mutable arguments (slower than Memoize)
#     """
#     def __init__(self, fn):
#         self.fn = fn
#         self.memo = {}
#     def __call__(self, *args):
#         import pickle
#         str = pickle.dumps(args)
#         if not self.memo.has_key(str):
#             self.memo[str] = self.fn(*args)
#         return self.memo[str]


# get_points = MemoizeMutable(get_points)


class CartesianPoint:
    """A class to hold a point's data."""
    BANK = {}

    def __new__(cls, *args, **kwargs):
        if kwargs:
            xys = (*args, kwargs['board_length'])
        else:
            xys = args
        if xys not in cls.BANK:
            # logger.info(f"__new__: A case BANK={cls.BANK}")
            return super().__new__(cls)
        # logger.info(f"__new__: B case BANK={cls.BANK}")
        return cls.BANK[xys]

    def __init__(self, x: int, y: int, board_length: int = 48):
        trix = (x, y, board_length)
        if trix not in self.BANK:
            # logger.info("entering CartesianPoint if block from __init__")
            self.x = x
            self.y = y
            self.xy = (x, y)
            self.board_length = board_length
            self.lookup = SPACE_LOOKUP[INVERSE[(x, y, board_length)]]

            self.at_right_edge = self.lookup['at_right_edge']
            self.at_left_edge = self.lookup['at_left_edge']
            self.at_top_edge = self.lookup['at_top_edge']
            self.at_bottom_edge = self.lookup['at_bottom_edge']
            self.BANK[trix] = self

    @property
    def all_neighbors(self):
        return get_points(self, 'all_neighbors')

    @property
    def surrounding_neighbors(self):
        return get_points(self, 'surrounding_neighbors')

    @property
    def plant_first_lichen_tiles(self):
        return get_points(self, 'plant_first_lichen_tiles')

    def __hash__(self):
        return hash((self.x, self.y, self.board_length))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def top_neighbor(self): return CartesianPoint(
        self.x, self.y - 1, self.board_length)

    @property
    def bottom_neighbor(self): return CartesianPoint(
        self.x, self.y + 1, self.board_length)

    @property
    def left_neighbor(self): return CartesianPoint(
        self.x - 1, self.y, self.board_length)

    @property
    def right_neighbor(self): return CartesianPoint(
        self.x + 1, self.y, self.board_length)

    @property
    def top_left_neighbor(self): return self.top_neighbor.left_neighbor
    @property
    def bottom_left_neighbor(self): return self.bottom_neighbor.left_neighbor
    @property
    def top_right_neighbor(self): return self.top_neighbor.right_neighbor
    @property
    def bottom_right_neighbor(self): return self.bottom_neighbor.right_neighbor

    # @property  # TODO: SLOW
    # def all_neighbors(self):
    #     neighbors = set()
    #     if not self.at_top_edge:
    #         neighbors.update({self.top_neighbor})
    #     if not self.at_bottom_edge:
    #         neighbors.update({self.bottom_neighbor})
    #     if not self.at_right_edge:
    #         neighbors.update({self.right_neighbor})
    #     if not self.at_left_edge:
    #         neighbors.update({self.left_neighbor})
    #     return neighbors

    # @property
    # def surrounding_neighbors(self):
    #     neighbors = set()
    #     if not self.at_top_edge:
    #         neighbors.update({self.top_neighbor})
    #         if not self.at_right_edge:
    #             neighbors.update({self.top_right_neighbor})
    #         if not self.at_left_edge:
    #             neighbors.update({self.top_left_neighbor})
    #     if not self.at_bottom_edge:
    #         neighbors.update({self.bottom_neighbor})
    #         if not self.at_right_edge:
    #             neighbors.update({self.bottom_right_neighbor})
    #         if not self.at_left_edge:
    #             neighbors.update({self.bottom_left_neighbor})
    #     if not self.at_right_edge:
    #         neighbors.update({self.right_neighbor})
    #     if not self.at_left_edge:
    #         neighbors.update({self.left_neighbor})
    #     return neighbors

    # @property
    # def plant_first_lichen_tiles(self):
    #     tiles = set()
    #     pre_tiles = self.surrounding_neighbors
    #     for pre_tile in pre_tiles:
    #         for n in pre_tile.all_neighbors:
    #             if n not in pre_tiles:
    #                 tiles.update({n})
    #     return tiles

    def __repr__(self):
        return "x:{} y:{} size:{}".format(self.x, self.y, self.board_length)

    def __str__(self):
        return "x:{} y:{}".format(self.x, self.y)


@functools.lru_cache(maxsize=3*48*48)
def get_points(point_key: CartesianPoint, dkey: str):
    bl = point_key.board_length
    l = point_key.lookup
    return set(CartesianPoint(*v, bl) for v in l[dkey])


class ConnectedComponent:
    def __init__(self, iterable: Iterator[CartesianPoint]):
        self.content = frozenset(iterable)
        self.area = len(self.content)

    def __len__(self):
        return len(self.content)

    def __eq__(self, other):
        return self.content == other.content

    def __hash__(self):
        return hash(self.content)

    def __repr__(self):
        return 'connected component=' + repr(self.content)

    def __iter__(self):
        return iter(self.content)

    def touches_point(self, point: CartesianPoint):
        if point in self.content:
            return True
        for inset in self.content:  # loop through existing tiles
            if point in inset.all_neighbors:
                return True
        return False

    @staticmethod
    def union(components):
        return ConnectedComponent(
            frozenset.union(*[c.content for c in components]))


if __name__ == "__main__":
    pass
