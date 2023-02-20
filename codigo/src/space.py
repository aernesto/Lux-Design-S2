# -*- coding: utf-8 -*-
from collections import namedtuple
BasePoint = namedtuple("BasePoint", "x y", module=__name__)


class CartesianPoint(BasePoint):
    """A namedtuple subclass to hold a point's data."""
    __slots__ = ()

    # next three properties set with pattern found here:
    # https://stackoverflow.com/a/5718537
    # not sure this is best practice...
    def get_board_length(self, length: int = 48):
        return length

    #  board_length = property(get_board_length)

    def get_at_bottom_edge(self, length: int = 48):
        return self.y == self.get_board_length(length) - 1

    #  at_bottom_edge = property(get_at_bottom_edge)

    def get_at_right_edge(self, length: int = 48):
        return self.x == self.get_board_length(length) - 1

    #  at_right_edge = property(get_at_right_edge)

    def __repr__(self):
        return "x: {}, y: {}".format(self.x, self.y)

    @property
    def at_top_edge(self):
        return self.y == 0

    @property
    def at_left_edge(self):
        return self.x == 0

    @property
    def top_neighbor(self):
        return CartesianPoint(self.x, self.y - 1)

    @property
    def bottom_neighbor(self):
        return CartesianPoint(self.x, self.y + 1)

    @property
    def left_neighbor(self):
        return CartesianPoint(self.x - 1, self.y)

    @property
    def right_neighbor(self):
        return CartesianPoint(self.x + 1, self.y)


if __name__ == "__main__":
    pass
