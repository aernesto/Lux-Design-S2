# -*- coding: utf-8 -*-
import logging
from obs import FactoryCenteredObservation, CenteredObservation
from sklearn.mixture import GaussianMixture
from luxai_s2.env import EnvConfig
import numpy as np
from collections import namedtuple
from typing import Sequence, Optional
from space import identify_conn_components, CartesianPoint
from robots import MapPlanner

Array = np.ndarray


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


class GmmMapSpawner:
    def __init__(self, obs: CenteredObservation):
        self.original_obs = obs
        self.total_factories = self.original_obs.my_team['factories_to_place']
        self.original_density = self.train_three_gmms(self.original_obs,
                                                      rubble_coef=1,
                                                      ice_coef=0,
                                                      ore_coef=0)

    def train_three_gmms(self,
                         obs: CenteredObservation,
                         rubble_coef: float = .9,
                         ice_coef: float = .05,
                         ore_coef: float = .05):
        # ICE
        ice = obs.ice_map
        ice_train = np.vstack(ice.nonzero()).T
        # TODO: include logic to avoid spawning too much in same area
        # TODO: the 'factories_to_place' below currently changes at each call
        clf_ice = GaussianMixture(
            n_components=len(ice_train),
            covariance_type="spherical",
            init_params='kmeans',
            means_init=ice_train,
        )
        clf_ice.fit(ice_train)

        # ORE
        ore = obs.ore_map
        ore_train = np.vstack(ore.nonzero()).T
        clf_ore = GaussianMixture(
            n_components=len(ore_train),
            covariance_type="spherical",
            init_params='kmeans',
            means_init=ore_train,
        )
        clf_ore.fit(ore_train)

        # RUBBLE
        rubble = obs.rubble_map.copy()
        threshold = 0  # TODO: don't hard-code this value
        rubble[rubble <= threshold] = 0
        rubble_train = np.vstack(np.where(rubble == 0)).T
        if len(rubble_train) == 0:
            rubble_train = np.vstack(np.where(rubble <= 5)).T

        clf_rubble = GaussianMixture(
            n_components=len(rubble_train),
            covariance_type="spherical",
            init_params='kmeans',
            means_init=rubble_train,
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


class ConnCompMapSpawner:
    def __init__(self,
                 obs: CenteredObservation,
                 threshold: float = 0,
                 rad: float = 30):
        self.rad = rad
        self.original_obs = obs
        self.obs_ = None  # will get set by choose_spawn_loc
        self.planner = MapPlanner(self.original_obs)
        self.rubble = self.planner.rubble
        self.board_length = len(self.rubble)
        self.total_factories = self.original_obs.my_team['factories_to_place']
        self.thr = threshold
        self.min_self_distance = 400
        self.self_avoidance_reward = 20
        self.resource_score_coef = 2
        self.components = identify_conn_components(self.rubble, self.thr)
        self.min_lichen_tiles = {
            0: 10,
            1: 8,
            2: 6,
            3: 6,
            4: 6,
            5: 4  # a priori never used
        }

    def _get_potential_spawns(self):
        x, y = np.where(self.obs_.board["valid_spawns_mask"] == 1)
        inner_list = list(zip(x, y, [self.board_length] * len(x)))
        potential_spawns = [CartesianPoint(*x_) for x_ in inner_list]
        return x, y, potential_spawns

    def choose_spawn_loc(self, obs: Optional[CenteredObservation] = None):
        if obs is None:
            obs = self.original_obs
        self.obs_ = obs
        x, y, potential_spawns = self._get_potential_spawns()
        num_factories = len(self.obs_.my_factories)
        min_lichen = self.min_lichen_tiles[num_factories]
        scores, info = self.score(potential_spawns, min_lichen)
        dtype = [('score', float), ('x', int), ('y', int)]
        a = np.array(list(zip(scores, x, y)),
                     dtype=dtype)  # create a structured array
        sorted_ = np.flip(np.sort(a, order='score'))
        logging.debug('sorted_first_10={}'.format(sorted_[:10]))
        logging.debug('sorted_last_10={}'.format(sorted_[-10:]))
        try:
            xy_list = [sorted_[0]['x'], sorted_[0]['y']]
            selection = np.array(xy_list)
            res_info = info[CartesianPoint(*xy_list, self.board_length)]
            logging.debug('selection={}'.format(selection))
        except (IndexError, AttributeError):
            logging.debug('scores.shape={}'.format(scores.shape))
            logging.debug('x.shape={}'.format(x.shape))
            logging.debug('y.shape={}'.format(y.shape))
            logging.debug('a.shape={}'.format(a.shape))
            logging.debug('sorted_.shape={}'.format(sorted_.shape))
            logging.debug('sorted_[0]={}'.format(sorted_[0]))
            logging.debug("sorted_[0]['y']={}".format(sorted_[0]))
            raise
        return selection, res_info['ice_set'], res_info['ore_set']

    def score(
        self,
        points: Sequence[CartesianPoint],
        min_lichen_tiles: int
    ) -> Array:
        scores = []
        resource_info = {}
        for point in points:
            score = 0
            lichen_tiles = point.plant_first_lichen_tiles
            for p in lichen_tiles:
                # add 1 point per 0-rubble first lichen tile
                score += self.rubble[p.x, p.y] <= self.thr

            # if score above is below 6, skip
            if score < min_lichen_tiles:
                scores.append(0)
                continue

            # add resources score
            ice_count, ice_set, ore_count, ore_set = self.planner.resources_radial_count(point, self.rad)
            score += self.resource_score_coef * (ice_count + ore_count)
            resource_info[point] = {
                'ice_count': ice_count,
                'ice_set': ice_set,
                'ore_count': ore_count,
                'ore_set': ore_set
            }
            # if score < min_resource_score:
            #     scores.append(score)
            #     continue

            # add area score
            for c in self.components:
                if any(n in c.content for n in lichen_tiles):
                    score += c.area
                    break

            # add distance from self score
            for plant_id in self.obs_.my_factories:
                fac_obs = FactoryCenteredObservation(
                    self.obs_.dict_obj,
                    plant_id
                )
                if self.planner.heavy_distance(point, fac_obs.pos) > self.min_self_distance:
                    score += self.self_avoidance_reward

            # # penalize based on distance to resources
            # for resource_tile in self.obs_.resources_pos:
            #     if score > 0:
            #         score -= self.planner.heavy_distance(point, resource_tile)

            scores.append(score)
            logging.debug('point={} gets a score of {}'.format(point, score))
        return np.array(scores), resource_info


if __name__ == "__main__":
    pass
