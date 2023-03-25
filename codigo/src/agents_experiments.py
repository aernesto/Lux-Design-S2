# -*- coding: utf-8 -*-
from copy import deepcopy
import logging
from luxai_s2.state import State
from luxai_s2.env import EnvConfig, Board, LuxAI_S2
from typing import Optional, Mapping, Dict, Union
from space import CartesianPoint
from collections import OrderedDict, defaultdict
import numpy as np
from itertools import cycle
from luxai_s2.utils import animate, my_turn_to_place_factory
from obs import (CenteredObservation,
                 RobotId,
                 HEAVY_TYPE,
                 LIGHT_TYPE,
                 PlantAssignment,
                 PlantId)
from robots import MapPlanner, RobotEnacter
from plants import PlantEnacter, ConnCompMapSpawner, GmmMapSpawner
from codetiming import Timer
Array = np.ndarray
logger = logging.getLogger(__name__)


class ControlledAgent:
    def __init__(
        self,
        player: str,
        env_cfg: EnvConfig,
        enable_monitoring: bool = False,
        spawn_method: str = 'gmm',
        max_sample: int = None,
        **kwargs
    ) -> None:
        self.spawn_method = spawn_method
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg
        self.board_length: int = self.env_cfg.map_size
        self.max_sample = max_sample

        self.max_allowed_factories: int = None  # set in early_setup method
        self.planner: MapPlanner = None  # set in early_setup
        self.map_spawner = None  # set in early_setup

        self.oracle = self.initialize_oracle()

        self._rng = np.random.default_rng()

        self.enable_monitoring = enable_monitoring
        self.stats = []

        self.options = kwargs

        self.max_per_plant = {HEAVY_TYPE: 8, LIGHT_TYPE: 20}

        self.spawn_timer = Timer(
            f"{self.player} spawn_timer", text="Generate Spawners: {:.2f}", logger=logger.info)
        self.choose_loc_timer = Timer(
            f"{self.player} choose_loc_timer", text="Choose Locations: {:.2f}", logger=logger.info)
        
        self.tile_iterator = defaultdict(dict)
        self.resource_iter = defaultdict(dict)

    def initialize_oracle(self):
        return defaultdict(dict, {
            'heavy_price': {
                'metal': self.env_cfg.ROBOTS['HEAVY'].METAL_COST,
                'power': self.env_cfg.ROBOTS['HEAVY'].POWER_COST
            },
            'light_price': {
                'metal': self.env_cfg.ROBOTS['LIGHT'].METAL_COST,
                'power': self.env_cfg.ROBOTS['LIGHT'].POWER_COST
            },
            'rubble': np.zeros((self.board_length, self.board_length), int)
        })

    def _tile_iterator(self, center: CartesianPoint):
        """Create a generator that cycles over surrounding neighbors."""
        def generator():
            point = {
                0: center.top_neighbor,
                1: center.top_right_neighbor,
                2: center.right_neighbor,
                3: center.bottom_right_neighbor,
                4: center.bottom_neighbor,
                5: center.bottom_left_neighbor,
                6: center.left_neighbor,
                7: center.top_left_neighbor
            }
            i = 0
            while True:
                i += 1
                yield point[i % 8]
        return generator()

    def _resource_iterator(self, points):
        """Create a cycle generator on points."""
        return cycle(points)

    def _update_plant_info(self, obs: CenteredObservation):
        """Update the oracle at each call of act() method.

        Side Effects:
          - modifies 'robots_to_plants_quotas' entry of oracle attr.
          - modifies resource_iter attr.

        Args:
          obs (CenteredObservation): current observation.        
        """
        # for each factory:
        for plant_id, props in obs.my_factories.items():
            fac_id = PlantId(plant_id, CartesianPoint(
                *props['pos'], self.board_length))
            
            # create its robot quotas
            if fac_id not in self.oracle['robots_to_plants_quotas']:
                self.oracle['robots_to_plants_quotas'][fac_id] = {
                    HEAVY_TYPE: {'current': 0, 'max': self.max_per_plant[HEAVY_TYPE]},
                    LIGHT_TYPE: {'current': 0,
                                 'max': self.max_per_plant[LIGHT_TYPE]}
                }

                # create its tile and resource iterators
                self.tile_iterator[fac_id] = self._tile_iterator(fac_id.pos)
                ice_nghb = self.oracle['plant_resource_nghb'][fac_id.pos]['ice']
                ore_nghb = self.oracle['plant_resource_nghb'][fac_id.pos]['ore']
                self.resource_iter[fac_id] = {
                    'ice': self._resource_iterator(ice_nghb),
                    'ore': self._resource_iterator(ore_nghb),
                }
        # TODO: clean up dead plants and figure out what to do with assigned units

    def _update_rubble_map(self):
        """Update 'rubble' entry of oracle attr."""
        old = self.oracle['rubble'].copy()
        new = self.oracle['obs'].rubble_map.copy()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Total rubble diff ={}".format((old - new).sum()))
        # TODO: do something here?
        self.oracle['rubble'] = new

    def update_oracle(self, obs: CenteredObservation):
        """Update oracle at each call of act method.
        
        Side Effects:
          - modifies 'dobs' entry of oracle attr
          - modifies 'obs' entry of oracle attr
          - modifies 'planned_actions' entry of oracle attr.
          - calls _update_plant_info method:
            - modifies 'robots_to_plants_quotas' entry of oracle attr.
            - modifies resource_iter attr.
          - calls _update_rubble_map method:
            - modifies 'rubble' entry of oracle attr.
          - calls assign_and_count_robots method:
            - modifies 'robot_assignments_to_plants' entry of oracle attr.
            - modifies 'robot_to_resource' entry of oracle attr.
            - calls increment_robot_load method: 
                - modifies 'robots_to_plants_quotas' entry of oracle attr.
        """
        self._update_plant_info(obs)
        self.oracle['dobs'] = obs.dict_obj
        self.oracle['obs'] = obs
        self._update_rubble_map()
        # old_plan = deepcopy(self.oracle['planned_actions'])
        self.oracle['planned_actions'] = dict()  # reset action plan
        fac_ids = self.oracle['obs'].factory_ids

        # review assignment of units to plants
        old_assignment = self.oracle['robot_assignments_to_plants'].copy()
        # just a pointer
        new_assignment = self.oracle['robot_assignments_to_plants']

        # first, drop dead robots
        for dead_robot in set(old_assignment.keys()) - set(self.oracle['obs'].robot_ids):
            self.assign_and_count_robots(
                dead_robot, old_assignment[dead_robot], -1)

        # then assign robots to plants as needed
        for robot in self.oracle['obs'].generate_robot_obs():
            rid = robot.myself
            if rid not in new_assignment:
                # check if robot was just created, in which case assign to mother plant
                # for now I check this by equating robot pos to plant pos
                # note that in theory, if robot was created, factory plant quota was not met last turn
                for plant in fac_ids:
                    if plant.pos == robot.pos:
                        tile = PlantAssignment(
                            plant.unit_id, plant.pos, next(self.tile_iterator[plant]))
                        self.assign_and_count_robots(rid, tile, 1)
                        break
                if rid in new_assignment:
                    continue
                # otherwise assign to closest "not full" factory for now
                forbidden = []
                for fac_id in fac_ids:
                    if not self.unmet_robot_quota(fac_id, robot.my_type):
                        forbidden.append(fac_id)
                close_to_far: Array = self.planner.rank_factories_by_distance_from(
                    robot.pos, cost_type=robot.cost_type)

                # find closest not full
                for fac_info in [f for f in close_to_far if f['fac'] not in forbidden]:
                    # TODO: is there a clever way to use fac.tile here?
                    fac = fac_info['fac']
                    tile = PlantAssignment(
                        fac.unit_id, fac.pos, next(self.tile_iterator[fac]))
                    self.assign_and_count_robots(rid, tile, 1)
                    break

                # if all factories full, issue warning and assign to closest full one
                if rid in new_assignment:
                    continue

                for fac_info in [f for f in close_to_far if f['fac'] in forbidden]:
                    fac = fac_info['fac']
                    tile = PlantAssignment(
                        fac.unit_id, fac.pos, next(self.tile_iterator[fac]))
                    self.assign_and_count_robots(rid, tile, 1)
                    break

    def _bernoulli(self, p):
        return self._rng.random() < p

    def assign_and_count_robots(self, robotid: RobotId, plant: PlantAssignment, increment: int) -> None:
        """Bookkeeping of oracle regarding robot assignment to plant and resource type.

        Side Effects:
          - modifies 'robot_assignments_to_plants' entry of oracle attr.
          - modifies 'robot_to_resource' entry of oracle attr.
          - calls increment_robot_load method: 
               - modifies 'robots_to_plants_quotas' entry of oracle attr.
          
        Args:
            robotid (RobotId): robot to be assigned or removed
            plant (PlantAssignment): plant receiving the robot assignment (or removal) instruction
            increment (int): +1 to assign the robot and -1 to remove it

        Raises:
            ValueError: If increment is neither 1 or -1.
        """
        assgn = self.oracle['robot_assignments_to_plants']
        resource_assgn = self.oracle['robot_to_resource']
        if increment == 1:
            # assign robot to plant
            assgn[robotid] = plant

            # assign resource type to robot
            # TODO: better logic below?
            resource_assgn[robotid] = 'ice' if self._bernoulli(.7) else 'ore'
        elif increment == -1:
            assgn.pop(robotid)
        else:
            raise ValueError("increment arg must be -1 or 1")

        # udpate the plants quotas dict
        self.increment_robot_load(plant, robotid, increment)

    def unmet_robot_quota(self, facid: PlantId, robot_type: str) -> bool:
        vals = self.oracle['robots_to_plants_quotas'][facid][robot_type]
        return vals['current'] < vals['max']

    def increment_robot_load(
            self,
            fac: Union[PlantAssignment, PlantId],
            robot_id: RobotId,
            by_value: int = 1
    ) -> None:
        """Updates the counts stored in oracle attr.

        Side Effects:
          - modifies 'robots_to_plants_quotas' entry of oracle attr.

        Args:
            fac (Union[PlantAssignment, PlantId]): Factory, the record of which to modify.
            robot_id (RobotId): Robot, the type of which is being tracked.
            by_value (int, optional): Count delta to apply. Defaults to 1.
        """
        facid = PlantId(fac.unit_id, fac.pos)
        self.oracle['robots_to_plants_quotas'][facid][robot_id.type]['current'] += by_value

    def factory_actions(self, step):
        # decide how many heavy robots to build
        # current logic is to build 8 heavy robots per factory
        obs = self.oracle['obs']
        actions = self.oracle['planned_actions']

        # total_heavy_quota = obs.num_factories * 8
        # existing_heavies = len(obs.my_heavy_units)
        # num_heavy_to_build = total_heavy_quota - existing_heavies

        for fac in obs.generate_factory_obs():
            plant_enacter = PlantEnacter(fac, self.env_cfg)
            if self.unmet_robot_quota(fac.myself, HEAVY_TYPE):
                enough_metal = fac.metal >= self.oracle['heavy_price']['metal']
                enough_power = fac.power >= self.oracle['heavy_price']['power']
                if enough_metal and enough_power:
                    # TODO: add logic to avoid spawning if robot present on center tile
                    actions[fac.unit_id] = plant_enacter.build_heavy()
            # TODO: deal with light robots

            if fac.unit_id not in actions:
                # TODO: better estimate cost of watering
                if (fac.water > 800) or (step > 800):
                    actions[fac.unit_id] = plant_enacter.water()
        return actions

    def robot_actions(self):
        obs = self.oracle['obs']
        actions = self.oracle['planned_actions']
        for robot in obs.generate_robot_obs():
            if not robot.queue_is_empty:
                # TODO: add logic to change queue if necessary
                continue
            # this robot's queue is empty
            # we assume robot is already assigned to a plant, if not it's a bug
            assigned_plant = self.oracle['robot_assignments_to_plants'][robot.myself]
            plant = PlantId(assigned_plant.unit_id, assigned_plant.pos)
            # Decide what the robot will do; a few options are:
            # 1. mine ice
            # 2. mine ore
            # 3. dig rubble
            # 4. dig ennemy's lichen
            # 5. attack ennemy robot
            # 6. self-destruct at location
            # 7. pickup resource or power from plant
            # 8. transfer resource or power to plant or robot

            # for now assign to ice and ore cycles
            enacter = RobotEnacter(robot, self.env_cfg)

            try:
                resource_type = self.oracle['robot_to_resource'][robot.myself]
                target_tile = next(self.resource_iter[plant][resource_type])
            except StopIteration:
                # try other resource
                self._flip_robot_resource_assignment(robot.myself)
                try:
                    resource_type = self.oracle['robot_to_resource'][robot.myself]
                    target_tile = next(
                        self.resource_iter[plant][resource_type])
                except StopIteration:
                    # this is not going well
                    # TODO: let's make this robot a Kamikaze?
                    logger.warning(f"robot {robot.myself} doesn't know what to do.")
                    pass
                else:
                    actions[robot.unit_id] = enacter.dig_cycle(
                        target_tile,
                        resource_type,
                        assigned_plant.tile,
                        dig_n=5
                    )
            else:
                actions[robot.unit_id] = enacter.dig_cycle(
                    target_tile,
                    resource_type,
                    assigned_plant.tile,
                    dig_n=5
                )

    def _flip_robot_resource_assignment(self, rid: RobotId):
        storage = self.oracle['robot_to_resource']
        old = storage[rid]
        storage[rid] = 'ice' if old == 'ore' else 'ore'

    def early_setup(self, step: int, dobs: Mapping, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            try:
                with self.spawn_timer:
                    obs_ = CenteredObservation(dobs, self.player)
                    # gets overwritten intentionally, so we stick with the last
                    self.planner = MapPlanner(obs_)
                    if self.spawn_method == 'conn':
                        self.map_spawner = ConnCompMapSpawner(
                            obs_,
                            self.planner,
                            gen=self._rng,
                            threshold=self.options['threshold'],
                            rad=self.options['radius'],
                        )
                    elif self.spawn_method == 'gmm':
                        self.map_spawner = GmmMapSpawner(
                            obs_,
                            self.planner,
                            rad=self.options['radius'],
                            gen=self._rng
                        )
                    else:
                        raise ValueError("unknown spawn_method")
            except KeyError:
                logger.error('options={}'.format(self.options))
                raise

            myteam = dobs['teams'][self.player]
            factories_to_place = myteam['factories_to_place']
            if self.max_allowed_factories is None:
                self.max_allowed_factories = factories_to_place
            if self.enable_monitoring:
                self.monitor(step, dobs)

            my_turn_to_place = my_turn_to_place_factory(
                myteam['place_first'], step)

            # TODO: add logic to control metal and water allocation
            if factories_to_place > 0 and my_turn_to_place:
                with self.choose_loc_timer:
                    spawn_loc, ice, ore = self.map_spawner.choose_spawn_loc(self.max_sample)

                center = CartesianPoint(*spawn_loc)
                odict = self.oracle['plant_resource_nghb']
                # breakpoint()
                odict[center] = {'ice': ice, 'ore': ore}
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, dobs: Mapping, remainingOverageTime: int = 60) -> Dict:
        """Main function to control our actions.

        Args:
            step (int): time step in game
            dobs (Mapping): standard dict of observations
            remainingOverageTime (int, optional): Not sure. Defaults to 60.

        Returns:
            Dict: actions
        """
        if self.enable_monitoring:
            self.monitor(step, dobs)
        observation = CenteredObservation(dobs, self.player)
        self.update_oracle(observation)
        # breakpoint()
        self.factory_actions(step)  # will update self.oracle['planned_actions']
        # breakpoint()
        self.robot_actions()  # will update self.oracle['planned_actions']
        # breakpoint()
        return self.oracle['planned_actions']

    def monitor(self, step, dobs):
        o = CenteredObservation(dobs, self.player)
        assertion_msg = "step type{} step val{} len val{}".format(
            type(step), step, len(self.stats))
        assert step == (len(self.stats) + 1), assertion_msg

        # factories stats
        fc = deepcopy(o.total_factories_cargo)
        tot_factory_power = np.sum(o.factories_ranked_by_power['power'])
        fc.update({
            'max_number_allowed': self.max_allowed_factories,
            'number_existing': len(o.my_factories),
            'power': tot_factory_power,
        })

        # Robots stats
        rc = deepcopy(o.total_robots_cargo)

        def sum_robot_power(dict_iterator):
            return sum(v['power'] for v in dict_iterator)

        tot_robot_power = sum_robot_power(o.my_units.values())
        rc.update({
            'max_number_allowed': self.max_allowed_factories,
            'number_existing': len(o.my_units),
            'power': tot_robot_power,
        })

        # TODO: track total lichen
        general = {
            'total_power_in_game': tot_factory_power + tot_robot_power,
        }

        self.stats.append(
            {'factories_total': fc, 'robots_total': rc, 'general': general})


def reset_w_custom_board(environment: LuxAI_S2, seed: int,
                         custom_board: Board):
    environment.agents = environment.possible_agents[:]
    environment.env_steps = 0
    if seed is not None:
        environment.seed_val = seed
        environment.seed_rng = np.random.RandomState(seed=seed)
    else:
        environment.seed_val = np.random.randint(0, 2**32 - 1)
        environment.seed_rng = np.random.RandomState(seed=environment.seed_val)

    environment.state: State = State(
        seed_rng=environment.seed_rng,
        seed=environment.seed_val,
        env_cfg=environment.state.env_cfg,
        env_steps=0,
        board=custom_board,
    )

    environment.max_episode_length = environment.env_cfg.max_episode_length
    for agent in environment.possible_agents:
        environment.state.units[agent] = OrderedDict()
        environment.state.factories[agent] = OrderedDict()
        # if environment.collect_stats:
        #     environment.state.stats[agent] = create_empty_stats()
    obs = environment.state.get_obs()
    observations = {agent: obs for agent in environment.agents}
    return observations, environment


class IdleAgent(ControlledAgent):
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        return actions


def interact(env,
             agents,
             steps,
             animate_: str = '',
             break_at_first_action: bool = False,
             debug: bool = False,
             custom_board: Optional[Board] = None,
             seed=42):
    # reset our env
    if custom_board is None:
        obs = env.reset(seed)
    else:
        obs, env = reset_w_custom_board(env,
                                        seed=seed,
                                        custom_board=custom_board)
    np.random.seed(0)
    imgs = []
    step = 0

    interact_timer = Timer(
        f"interact_timer", text="single step (2 players) took: {:.2f}", logger=logger.info)
    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps:
            break
        actions = {}
        interact_timer.start()
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1

        obs, rewards, dones, infos = env.step(actions)
        interact_timer.stop()
        if step == 1:
            first_obs = deepcopy(obs)
        imgs += [env.render("rgb_array", width=640, height=640)]
        if debug:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('{step}'.format(step=step))
        if logger.isEnabledFor(logging.INFO):
            logger.info("Total time stats:{}".format(Timer.timers))

    done = False

    inner_counter = 0
    while not done:
        if step >= steps:
            break
        actions = {}
        interact_timer.start()
        for player in env.agents:
            o = obs[player]
            if debug:
                a = agents[player].debug_act(step, o)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("{step}  {inner_counter}".format(
                        step=step, inner_counter=inner_counter))
            else:
                a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        interact_timer.stop()
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]
        inner_counter += 1
        if break_at_first_action and inner_counter == 2:
            break
        if logger.isEnabledFor(logging.INFO):
            logger.info("Total time stats:{}".format(Timer.timers))
    if animate_:
        if logger.isEnabledFor(logging.INFO):
            logger.info('writing {animate_}'.format(animate_=animate_))
        return animate(imgs, filename=animate_)
    else:
        return first_obs, obs


def step_interact(env, agents, steps, map_seed):
    # reset our env
    obs = env.reset(seed=map_seed)
    np.random.seed(0)
    step = 0

    while env.state.real_env_steps < 0:
        actions = {}
        for player in env.agents:
            o = obs[player]
            actions[player] = agents[player].early_setup(step, o)
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Total time stats:{}".format(Timer.timers))
        yield obs, rewards, dones, infos, step, agents

    while step < steps:
        actions = {}
        for player in env.agents:
            o = obs[player]
            actions[player] = agents[player].act(step, o)
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Total time stats:{}".format(Timer.timers))
        yield obs, rewards, dones, infos, step, agents


if __name__ == "__main__":
    """
    $ python -m cProfile -o prof030623_1.prof agents_experiments.py <seed> <len> <logfile> <log level>
    """
    import sys

    # logger's optimization: https://docs.python.org/3.7/howto/logging.html#optimization
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0

    seed = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    fname = sys.argv[3]
    logging_level = sys.argv[4]
    ll = logging.INFO if logging_level.strip().lower() == 'info' else logging.WARNING
    logging.basicConfig(filename=fname, level=ll,
                        filemode='w')  # w filemode overwrites
    # make a random env
    env = LuxAI_S2()
    # obs = env.reset(seed=seed)
    agent0 = ControlledAgent('player_0', env.env_cfg,
                             spawn_method='conn', radius=130, threshold=15, max_sample=10)
    agent1 = ControlledAgent('player_1', env.env_cfg,
                             spawn_method='conn', threshold=15, radius=130, max_sample=20)
    interact(env, {'player_0': agent0, 'player_1': agent1},
             num_steps, seed=seed)
