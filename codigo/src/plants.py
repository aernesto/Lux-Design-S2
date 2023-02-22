# -*- coding: utf-8 -*-
import logging
from obs import FactoryCenteredObservation, CenteredObservation
from sklearn.mixture import GaussianMixture
from luxai_s2.env import EnvConfig
import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.items()}


_PLANT_ACTIONS = {
    0: 'build light',
    1: 'build heavy',
    2: 'water',
}
PLANT_ACTIONS = invert_dict(_PLANT_ACTIONS)


class PlantEnacter:
    def __init__(self, fac_obs: FactoryCenteredObservation,
                 env_cfg: EnvConfig):
        self.obs = fac_obs
        self.conf = env_cfg

    def build_heavy(self):
        return PLANT_ACTIONS['build heavy']

    def build_light(self):
        return PLANT_ACTIONS['build light']

    def water(self):
        return PLANT_ACTIONS['water']


class MapSpawner:
    def __init__(self, obs: CenteredObservation):
        self.original_obs = obs
        self.total_factories = self.original_obs.my_team['factories_to_place']
        self.original_density = self.train_three_gmms(self.original_obs)

    def train_three_gmms(self,
                         obs: CenteredObservation,
                         rubble_coef: float = .9,
                         ice_coef: float = .05,
                         ore_coef: float = .05):
        # ICE
        ice = obs.ice_map
        ice_train = np.vstack(ice.T.nonzero()).T
        # TODO: include logic to avoid spawning too much in same area
        # TODO: the 'factories_to_place' below currently changes at each call
        clf_ice = GaussianMixture(
            n_components=self.total_factories,
            covariance_type="spherical",
            init_params='kmeans',
        )
        clf_ice.fit(ice_train)

        # ORE
        ore = obs.ore_map
        ore_train = np.vstack(ore.T.nonzero()).T
        clf_ore = GaussianMixture(
            n_components=self.total_factories,
            covariance_type="spherical",
            init_params='kmeans',
        )
        clf_ore.fit(ore_train)

        # RUBBLE
        rubble = obs.rubble_map.copy()
        rubble[ice.nonzero()] = 0
        threshold = 0  # TODO: don't hard-code this value
        rubble[rubble <= threshold] = 0
        rubble_train = np.vstack(np.where(rubble.T == 0)).T
        clf_rubble = GaussianMixture(
            n_components=self.total_factories,
            covariance_type="spherical",
            init_params='kmeans',
        )
        clf_rubble.fit(rubble_train)

        # combine three densities
        def _d(samples):
            return (-ice_coef * clf_ice.score_samples(samples) -
                    ore_coef * clf_ore.score_samples(samples) -
                    rubble_coef * clf_rubble.score_samples(samples))

        return _d

    def choose_spawn_loc(self, obs: CenteredObservation):
        # TODO: include logic to avoid spawning too much in same area
        # TODO: the 'factories_to_place' below currently changes at each call
        inner_list = list(zip(*np.where(obs.board["valid_spawns_mask"] == 1)))
        potential_spawns = np.array(inner_list)
        # lower score is better
        scores = self.original_density(potential_spawns)
        x = potential_spawns[:, 0]
        y = potential_spawns[:, 1]
        dtype = [('score', float), ('x', int), ('y', int)]
        all_factories = []
        a = np.array(list(zip(scores, x, y)),
                     dtype=dtype)  # create a structured array
        sorted_ = np.sort(a, order='score')
        try:
            return np.array([sorted_[0]['x'], sorted_[0]['y']])
        except (IndexError, AttributeError):
            logging.debug('scores.shape={}'.format(scores.shape))
            logging.debug('x.shape={}'.format(x.shape))
            logging.debug('y.shape={}'.format(y.shape))
            logging.debug('a.shape={}'.format(a.shape))
            logging.debug('sorted_.shape={}'.format(sorted_.shape))
            logging.debug('sorted_[0]={}'.format(sorted_[0]))
            logging.debug("sorted_[0]['y']={}".format(sorted_[0]))
            raise


if __name__ == "__main__":
    pass
