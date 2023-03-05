import unittest
from space import CartesianPoint, ConnectedComponent, xy_iter
import sys
sys.path.append('..')


class TestSpace(unittest.TestCase):
    def setUp(self):
        self.x, self.y = 0, 3
        self.len = 48
        self.point = CartesianPoint(self.x, self.y, self.len)
        self.top = CartesianPoint(self.x, self.y - 1, self.len)
        self.right = CartesianPoint(self.x + 1, self.y, self.len)

    def test_xy_iter(self):
        points = {
            CartesianPoint(0, 0, 2),
            CartesianPoint(0, 1, 2),
            CartesianPoint(1, 0, 2),
            CartesianPoint(1, 1, 2),
        }
        self.assertTrue(all(point in points for point in xy_iter(2)))

    def test_cartesian_point(self):
        self.assertTrue(self.point.at_left_edge)
        self.assertFalse(self.point.at_right_edge)
        self.assertFalse(self.point.at_bottom_edge)
        self.assertFalse(self.point.at_top_edge)

        self.assertEqual(self.point.top_neighbor, self.top)
        self.assertEqual(self.top.bottom_neighbor, self.point)
        self.assertEqual(self.point.x, self.x)
        self.assertEqual(self.point.y, self.y)

        all_neighbors = self.point.all_neighbors
        surr_neighbors = self.point.surrounding_neighbors
        self.assertIn(self.top, surr_neighbors)
        self.assertIn(self.top, all_neighbors)

        self.assertEqual(len(all_neighbors), 3)
        self.assertEqual(len(surr_neighbors), 5)

        self.assertIn(self.top.top_neighbor,
                      self.point.plant_first_lichen_tiles)

    def test_conn_component(self):
        comp = ConnectedComponent([self.point, self.top])
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp.area, 2)
        self.assertFalse(comp.touches_point(self.right.right_neighbor))
        self.assertTrue(comp.touches_point(self.right))
        new = ConnectedComponent([self.right])
        self.assertEqual(len(new), 1)
        merged = ConnectedComponent.union([comp, new])
        self.assertEqual(len(merged), len(comp) + len(new))
        self.assertIn(self.right, merged)
        self.assertNotIn(self.right, comp)
        newcomp = ConnectedComponent([self.point, self.top])
        self.assertEqual(comp, newcomp)
        self.assertNotEqual(comp, new)


if __name__ == '__main__':
    unittest.main()
