import sys
sys.path.append('..')
from space import CartesianPoint
import unittest

class TestSpace(unittest.TestCase):
    def test_cartesian_point(self):
        x, y = 2, 3
        point = CartesianPoint(x, y, 48)
        top = CartesianPoint(x, y - 1)
        self.assertEqual(point.top_neighbor, top)
        self.assertEqual(point.x, x)
        self.assertEqual(point.y, y)


if __name__ == '__main__':   
    unittest.main()
