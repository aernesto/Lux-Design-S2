# -*- coding: utf-8 -*-
import logging
import itertools
from typing import Sequence, Iterator
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


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
        if r.T[point.x, point.y] <= threshold:  # 0-rubble tiles
            logging.debug('visited point={}'.format(point))
            logging.debug('neighbor set={}'.format(point.all_neighbors))
            logging.debug('current components set={}'.format(components))
            touched_components = [
                c for c in components if c.touches_point(point)
            ]
            if len(touched_components):
                logging.debug(touched_components)
                components = components - set(
                    touched_components)  # remove touched
                touched_components += [ConnectedComponent(
                    (point, ))]  # add the point
                merger = ConnectedComponent.union(
                    touched_components)  # merge all
                components.update({merger})  # add them back in
            else:
                components.update({ConnectedComponent([point])})
            logging.debug('new components set ={}'.format(components))

    return components


@dataclass(frozen=True)
class CartesianPoint:
    """A class to hold a point's data."""
    x: int
    y: int
    board_length: int = 48

    def __post_init__(self):
        assert 0 <= self.x < self.board_length
        assert 0 <= self.y < self.board_length

    @property
    def at_bottom_edge(self):
        return self.y == self.board_length - 1

    @property
    def at_right_edge(self):
        return self.x == self.board_length - 1

    @property
    def at_top_edge(self):
        return self.y == 0

    @property
    def at_left_edge(self):
        return self.x == 0

    @property
    def top_neighbor(self):
        return CartesianPoint(self.x, self.y - 1, self.board_length)

    @property
    def bottom_neighbor(self):
        return CartesianPoint(self.x, self.y + 1, self.board_length)

    @property
    def left_neighbor(self):
        return CartesianPoint(self.x - 1, self.y, self.board_length)

    @property
    def right_neighbor(self):
        return CartesianPoint(self.x + 1, self.y, self.board_length)

    @property
    def all_neighbors(self):
        neighbors = set()
        if not self.at_top_edge:
            neighbors.update({self.top_neighbor})
        if not self.at_bottom_edge:
            neighbors.update({self.bottom_neighbor})
        if not self.at_right_edge:
            neighbors.update({self.right_neighbor})
        if not self.at_left_edge:
            neighbors.update({self.left_neighbor})
        return neighbors

    def __repr__(self):
        return "x:{} y:{} size:{}".format(self.x, self.y, self.board_length)

    def __str__(self):
        return "x:{} y:{}".format(self.x, self.y)


class ConnectedComponent:
    def __init__(self, iterable: Iterator[CartesianPoint]):
        self.content = frozenset(iterable)

    def __len__(self):
        return len(self.content)

    def __eq__(self, other):
        return self.content == other.content

    def __hash__(self):
        return hash(self.content)

    def __repr__(self):
        return 'connected component=' + repr(self.content)

    def touches_point(self, point: CartesianPoint):
        if point in self.content:
            return True
        for inset in self.content:  # loop through existing tiles
            if point in inset.all_neighbors:
                return True
        return False

    @property
    def area(self):
        return len(self)

    @staticmethod
    def union(components):
        return ConnectedComponent(
            frozenset.union(*[c.content for c in components]))


if __name__ == "__main__":
    pass
