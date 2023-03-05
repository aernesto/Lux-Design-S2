import unittest
import sys
import numpy as np
sys.path.append('..')
from robots import (_move,
                    _DEFAULT_AMOUNT,
                    _DEFAULT_DIRECTION,
                    _DEFAULT_RESOURCE,
                    compress_queue,
                    flip_movement_queue
                    )

def arreq(a1, a2):
    return (a1 == a2).all()


class TestActions(unittest.TestCase):
    def setUp(self):
        self.actions = {}
        self.actions['up'] = {
            'n_5_repeat_0': np.array([
                0,  # move
                1,  # up
                _DEFAULT_RESOURCE,  # ice
                _DEFAULT_AMOUNT,  # amount
                0,  # repeat
                5,  # n
            ]),
            'n_2_repeat_0': np.array([
                0,  # move
                1,  # up
                _DEFAULT_RESOURCE,  # ice
                _DEFAULT_AMOUNT,  # amount
                0,  # repeat
                2,  # n
            ]),
            'n_1_repeat_4': np.array([
                0,
                1,
                _DEFAULT_RESOURCE,
                _DEFAULT_AMOUNT,
                4,
                1
            ]),
        }
        qrepeat_0 = [v for v in self.actions['up'].values() if v[4] == 0]
        sum_n_0 = sum(v[5] for v in qrepeat_0)

        qrepeat_4 = [v for v in self.actions['up'].values() if v[4] == 4]
        sum_n_4 = sum(v[5] for v in qrepeat_4)

        self.uncompressed_queue_repeat_0 = qrepeat_0
        self.uncompressed_queue_repeat_4 = qrepeat_4
        self.compressed_queue_repeat_0 = [
            np.array([0, 1, _DEFAULT_RESOURCE, _DEFAULT_AMOUNT, 0, sum_n_0])]
        self.compressed_queue_repeat_4 = [
            np.array([0, 1, _DEFAULT_RESOURCE, _DEFAULT_AMOUNT, 4, sum_n_4])]
        
        self.long_queue = [
            np.array([0, 1, 0, 0, 3, 3]), # up
            np.array([0, 1, 0, 0, 3, 3]), # up
            np.array([0, 2, 0, 0, 3, 3]), # right
            np.array([0, 3, 0, 0, 3, 3]), # down
            np.array([0, 3, 0, 0, 3, 3]), # down 
            np.array([0, 4, 0, 0, 3, 3]), # left - back to square 1
            np.array([0, 1, 0, 0, 3, 3]), # up - square 2
        ]
        self.inverse_long_queue = [
            np.array([0, 3, 0, 0, 3, 3]),  # down
            np.array([0, 2, 0, 0, 3, 3]),  # right
            np.array([0, 1, 0, 0, 3, 3]),  # up
            np.array([0, 1, 0, 0, 3, 3]),  # up
            np.array([0, 4, 0, 0, 3, 3]),  # left
            np.array([0, 3, 0, 0, 3, 3]),  # down
            np.array([0, 3, 0, 0, 3, 3]),  # down
        ]

    def test_move(self):
        exp = self.actions['up']['n_2_repeat_0']
        real = _move('up', 0, 2)
        self.assertTrue(arreq(exp, real))

        exp = self.actions['up']['n_1_repeat_4']
        real = _move('up', 4, 1)
        self.assertTrue(arreq(exp, real))

    def test_compress_queue(self):
        # breakpoint()
        real = compress_queue(self.uncompressed_queue_repeat_0)
        expected = self.compressed_queue_repeat_0
        for rea, exp in zip(real, expected):
            self.assertTrue(arreq(rea, exp))

        real = compress_queue(self.uncompressed_queue_repeat_4)
        expected = self.compressed_queue_repeat_4
        for rea, exp in zip(real, expected):
            # breakpoint()
            self.assertTrue(arreq(rea, exp))

    def test_inverse_queue(self):
        inverted = flip_movement_queue(self.long_queue)
        for rea, exp in zip(inverted, self.inverse_long_queue):
            self.assertTrue((rea == exp).all())

if __name__ == '__main__':
    unittest.main()
