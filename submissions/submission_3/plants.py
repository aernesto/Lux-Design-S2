# -*- coding: utf-8 -*-
import logging
from obs import FactoryCenteredObservation, CenteredObservation
from sklearn.mixture import GaussianMixture
from luxai_s2.env import EnvConfig
import numpy as np
from collections import namedtuple
from typing import Sequence, Optional, Tuple, Dict, Any, List
from space import identify_conn_components, CartesianPoint
from robots import MapPlanner
logger = logging.getLogger(__name__)
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


class MapSpawner:
    min_lichen_tiles = {
        0: 10,
        1: 8,
        2: 6,
        3: 6,
        4: 6,
        5: 4  # a priori never used
    }

    def __init__(self, obs: CenteredObservation, planner: MapPlanner, rad: float):
        self.obs = obs
        self.planner = planner
        self.rad = rad

    @property
    def board_length(self):
        return self.obs.board_length
    
    def score(
        self,
        points: Sequence[CartesianPoint],
        *args
    ) -> Tuple[Array, Dict]:
        raise NotImplementedError()

    def _get_potential_spawns(self):
        x, y = np.where(self.obs.board["valid_spawns_mask"] == 1)
        inner_list = list(zip(x, y, [self.board_length] * len(x)))
        potential_spawns = [CartesianPoint(*x_) for x_ in inner_list]
        return x, y, potential_spawns

    def choose_spawn_loc(self) -> Tuple[Array, Dict, Dict]:
        raise NotImplementedError()


class GmmMapSpawner(MapSpawner):
    def __init__(self, obs: CenteredObservation, planner: MapPlanner, rad: float):
        super().__init__(obs, planner, rad)
        self.total_factories = self.obs.my_team['factories_to_place']
        self.orig_density = self.train_three_gmms(self.obs,
                                                  rubble_coef=1,
                                                  ice_coef=0,
                                                  ore_coef=0)

    def score(self, points: Sequence[CartesianPoint], arr: List):
        resource_info = {}
        for point in points:
            ice_count, ice_set, ore_count, ore_set = self.planner.resources_radial_count(
                point, self.rad)
            resource_info[point] = {
                'ice_count': ice_count,
                'ice_set': ice_set,
                'ore_count': ore_count,
                'ore_set': ore_set
            }
        return self.orig_density(np.array(arr)), resource_info

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

    def choose_spawn_loc(self):
        # TODO: include logic to avoid spawning too much in same area
        # TODO: the 'factories_to_place' below currently changes at each call
        x, y, potential_spawns = self._get_potential_spawns()
        # lower score is better
        scores, info = self.score(potential_spawns, list(zip(x, y)))

        dtype = [('score', float), ('x', int), ('y', int)]
        a = np.array(list(zip(scores, x, y)),
                     dtype=dtype)  # create a structured array
        sorted_ = np.sort(a, order='score')
        try:
            xy_list = [sorted_[0]['x'], sorted_[0]['y']]
            selection = np.array(xy_list)
            # TODO: KeyError happened below
            res_info = info[CartesianPoint(*xy_list, self.board_length)]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('selection={}'.format(selection))
        except (IndexError, AttributeError):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('scores.shape={}'.format(scores.shape))
                logger.debug('x.shape={}'.format(x.shape))
                logger.debug('y.shape={}'.format(y.shape))
                logger.debug('a.shape={}'.format(a.shape))
                logger.debug('sorted_.shape={}'.format(sorted_.shape))
                logger.debug('sorted_[0]={}'.format(sorted_[0]))
                logger.debug("sorted_[0]['y']={}".format(sorted_[0]))
            raise
        return selection, res_info['ice_set'], res_info['ore_set']

class ConnCompMapSpawner(MapSpawner):
    def __init__(self,
                 obs: CenteredObservation,
                 planner: MapPlanner,
                 threshold: float = 0,
                 rad: float = 30):
        super().__init__(obs, planner, rad)
        self.total_factories = self.obs.my_team['factories_to_place']
        self.thr = threshold
        self.min_self_distance = 400
        self.self_avoidance_reward = 20
        self.resource_score_coef = 2
        self.components = identify_conn_components(self.rubble, self.thr)

    @property
    def rubble(self):
        return self.planner.rubble

    def choose_spawn_loc(self):
        x, y, potential_spawns = self._get_potential_spawns()
        num_factories = len(self.obs.my_factories)
        min_lichen = self.min_lichen_tiles[num_factories]
        scores, info = self.score(potential_spawns, min_lichen)
        dtype = [('score', float), ('x', int), ('y', int)]
        a = np.array(list(zip(scores, x, y)),
                     dtype=dtype)  # create a structured array
        sorted_ = np.flip(np.sort(a, order='score'))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('sorted_first_10={}'.format(sorted_[:10]))
            logger.debug('sorted_last_10={}'.format(sorted_[-10:]))

        try:
            xy_list = [sorted_[0]['x'], sorted_[0]['y']]
            selection = np.array(xy_list)
            # TODO: KeyError happened below
            res_info = info[CartesianPoint(*xy_list, self.board_length)]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('selection={}'.format(selection))
        except (IndexError, AttributeError):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('scores.shape={}'.format(scores.shape))
                logger.debug('x.shape={}'.format(x.shape))
                logger.debug('y.shape={}'.format(y.shape))
                logger.debug('a.shape={}'.format(a.shape))
                logger.debug('sorted_.shape={}'.format(sorted_.shape))
                logger.debug('sorted_[0]={}'.format(sorted_[0]))
                logger.debug("sorted_[0]['y']={}".format(sorted_[0]))
            raise
        return selection, res_info['ice_set'], res_info['ore_set']

    def score(
        self,
        points: Sequence[CartesianPoint],
        min_lichen_tiles: int
    ) -> Tuple[Array, Dict]:
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
            ice_count, ice_set, ore_count, ore_set = self.planner.resources_radial_count(
                point, self.rad)
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
            for plant_id in self.obs.my_factories:
                fac_obs = FactoryCenteredObservation(
                    self.obs.dict_obj,
                    plant_id
                )
                if self.planner.heavy_distance(point, fac_obs.pos) > self.min_self_distance:
                    score += self.self_avoidance_reward

            # # penalize based on distance to resources
            # for resource_tile in self.obs.resources_pos:
            #     if score > 0:
            #         score -= self.planner.heavy_distance(point, resource_tile)

            scores.append(score)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    'point={} gets a score of {}'.format(point, score))
        return np.array(scores), resource_info


if __name__ == "__main__":
    pass
