import sys
sys.path.append('..')
from obs import (RobotId,
                 PlantId,
                 PlantAssignment,
                 flip_name,
                 CenteredObservation,
                 RobotCenteredObservation,
                 FactoryCenteredObservation)
from space import CartesianPoint
from luxai_s2.env import LuxAI_S2
import unittest

import numpy as np



class TestCenteredObservation(unittest.TestCase):
    def setUp(self):
        self.player_name = 'player_0'
        self.env = LuxAI_S2()
        self.obs_dict = self.env.reset(seed=5)
        self.obs = CenteredObservation(
            self.obs_dict[self.player_name],
            self.player_name
        )

    def test_RobotId(self):
        rid = RobotId('unit_23', 'HEAVY')
        self.assertEqual(rid.type, 'HEAVY')
        self.assertEqual(rid.unit_id, 'unit_23')
        self.assertEqual(rid, RobotId('unit_23', 'HEAVY'))
        self.assertNotEqual(rid, RobotId('unit_23', 'LIGHTs'))

    def test_PlantId(self):
        pos = CartesianPoint(2, 3, 48)
        fname = 'factory_23'
        pid = PlantId(fname, pos)
        self.assertEqual(pid.pos, pos)
        self.assertEqual(pid.unit_id, fname)
        self.assertEqual(pid, PlantId(fname, pos))
        self.assertNotEqual(pid, PlantId(fname, pos.bottom_neighbor))
        self.assertEqual(len(pid), 2)

    def test_PlantAssignment(self):
        pos = CartesianPoint(2, 3, 48)
        fname = 'factory_23'
        pid = PlantAssignment(fname, pos, pos.right_neighbor)
        self.assertEqual(pid.pos, pos)
        self.assertEqual(pid.tile, pos.right_neighbor)
        self.assertEqual(pid.unit_id, fname)
        self.assertEqual(pid, PlantAssignment(fname, pos, pos.right_neighbor))
        self.assertNotEqual(
            pid,
            PlantAssignment(fname, pos.bottom_neighbor, pos.right_neighbor)
        )
        self.assertEqual(len(pid), 3)

    def test_flip_name(self):
        self.assertEqual(flip_name('player_0'), 'player_1')
        self.assertEqual(flip_name('player_1'), 'player_0')

    def test_centered_obs(self):
        obs = self.obs
        self.assertEqual(obs.myself, self.player_name)
        for tile in ['lichen', 'ice', 'ore', 'rubble']:
            self.assertIsInstance(obs.board[tile], np.ndarray)
        self.assertIsInstance(obs.ice_map, np.ndarray)
        self.assertIsInstance(obs.ore_map, np.ndarray)
        self.assertIsInstance(obs.rubble_map, np.ndarray)
        self.assertEqual(obs.ice_map.shape, (48, 48))
        self.assertTrue((obs.ice_map == obs.board['ice']).all())
        self.assertTrue(obs.ice_map.sum() > 2)
        self.assertEqual(obs.opponent, flip_name(self.player_name))


if __name__ == '__main__':
    unittest.main()
