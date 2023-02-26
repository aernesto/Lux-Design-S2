# -*- coding: utf-8 -*-
from copy import deepcopy
import logging
from luxai_s2.state import State
from luxai_s2.env import EnvConfig, Board, LuxAI_S2
from typing import Optional, Mapping, Dict
from space import CartesianPoint
from collections import OrderedDict, defaultdict
import numpy as np
from luxai_s2.utils import animate, my_turn_to_place_factory
from obs import CenteredObservation, RobotCenteredObservation, RobotId, HEAVY_TYPE, LIGHT_TYPE
from robots import RobotEnacter, MapPlanner
from plants import PlantEnacter, ConnCompMapSpawner, PlantAssignment


class ControlledAgent:
    def __init__(self, player: str, env_cfg: EnvConfig, **kwargs) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg
        self.max_allowed_factories: int = None  # set in early_setup method
        self.ice_pos: set = None  # set in early_setup
        self.ore_pos: set = None  # set in early_setup
        self.planner: MapPlanner = None  # set in early_setup
        self.map_spawner = None  # set in early_setup
        self.oracle = self.initialize_oracle()
        self.stats = []
        self.options = kwargs
        self.max_per_plant = {HEAVY_TYPE: 8, LIGHT_TYPE: 20}

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
        })

    def update_oracle_early_setup(self, dobs: Mapping):
        """Update the oracle at each call of early_setup."""
        # for each newly created factory, create its robot quotas
        for plant_id, props in dobs['factories'].items():
            fac_id = PlantAssignment(plant_id, props['pos'])
            if fac_id not in self.oracle['robots_to_plants_quotas']:
                self.oracle['robots_to_plants_quotas'][fac_id] = {
                    HEAVY_TYPE: {'current': 0, 'max': self.max_per_plant[HEAVY_TYPE]},
                    LIGHT_TYPE: {'current': 0,
                                 'max': self.max_per_plant[LIGHT_TYPE]}
                }

    def update_oracle(self, obs: CenteredObservation):
        """Update oracle at each call of act method."""
        self.oracle['dobs'] = obs.dict_obj
        self.oracle['obs'] = obs
        # old_plan = deepcopy(self.oracle['planned_actions'])
        self.oracle['planned_actions'] = dict()  # reset action plan
        fac_ids = self.oracle['obs'].factory_ids

        # review assignment of units to plants
        #TODO: think of how to manage actual target tiles for robots beyond plant centers
        old_assignment = self.oracle['robot_assignments_to_plants'].copy()
        # just a pointer
        new_assignment = self.oracle['robot_assignments_to_plants']
        plants_quotas = self.oracle['robots_to_plants_quotas']

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
                # note that in theory, if robot was created, factory plant was not met last turn
                for plant in fac_ids:
                    if plant.pos == robot.pos:
                        self.assign_and_count_robots(rid, plant, 1)
                        break

                # otherwise assign to closest "not full" factory for now
                forbidden = []
                for fac_id in fac_ids:
                    if not self.unmet_robot_quota(fac_id, robot.my_type):
                        forbidden.append(fac_id)
                close_to_far: list = self.planner.rank_factories_by_distance_from(
                    robot.pos, cost_type=robot.cost_type)

                # find closest not full
                for fac in [f['fac'] for f in close_to_far if f['fac'] not in forbidden]:
                    self.assign_and_count_robots(rid, fac, 1)
                    break

                # if all factories full, issue warning and assign to closest full one
                if rid not in new_assignment:
                    for fac in [f['fac'] for f in close_to_far if f['fac'] in forbidden]:
                        self.assign_and_count_robots(rid, fac, 1)
                        break

    def assign_and_count_robots(self, robotid: RobotId, plantid: PlantAssignment, increment: int) -> None:
        """Bookkeeping of oracle regarding robot assignment to plant.

        Args:
            robotid (RobotId): robot to be assigned or removed
            plantid (PlantAssignment): plant receiving the robot assignment (or removal) instruction
            increment (int): +1 to assign the robot and -1 to remove it

        Raises:
            ValueError: If increment is neither 1 or -1.
        """
        # update the robots to plants assignment dict
        assgn = self.oracle['robot_assignments_to_plants']
        if increment == 1:
            assgn[robotid] = plantid
        elif increment == -1:
            assgn.pop(robotid)
        else:
            raise ValueError("increment arg must be -1 or 1")

        # udpate the plants quotas dict
        self.increment_robot_load(plantid, robotid, increment)

    def unmet_robot_quota(self, facid: PlantAssignment, robot_type: str) -> bool:
        vals = self.oracle['robots_to_plants_quotas'][facid][robot_type]
        return vals['current'] < vals['max']

    def increment_robot_load(
            self,
            facid: PlantAssignment,
            robot_id: RobotId,
            by_value: int = 1
    ) -> None:
        self.oracle['robots_to_plants_quotas'][facid][robot_id.type]['current'] += by_value

    def build_robots(self):
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
                    actions[fac.unit_id] = plant_enacter.build_heavy()
        # TODO: deal with light robots
        return actions

    def water(self):
        actions = self.oracle['planned_actions']
        for fac in self.oracle['obs'].generate_factory_obs():
            plant_enacter = PlantEnacter(fac, self.env_cfg)
            if fac.unit_id not in actions:
                # TODO: estimate cost of watering
                actions[fac.unit_id] = plant_enacter.water()
        return actions

    def early_setup(self, step: int, dobs: Mapping, remainingOverageTime: int = 60):
        self.update_oracle_early_setup(dobs)
        if step == 0:
            ice_board = dobs['board']['ice']
            ore_board = dobs['board']['ore']
            self.ice_pos = set([
                CartesianPoint(x, y, len(ice_board))
                for x, y in zip(*ice_board.nonzero())
            ])
            self.ore_pos = set([
                CartesianPoint(x, y, len(ice_board))
                for x, y in zip(*ore_board.nonzero())
            ])
            #  logging.debug('{}'.format(self.ice_pos))
            return dict(faction="AlphaStrike", bid=0)
        else:
            try:
                self.map_spawner = ConnCompMapSpawner(
                    CenteredObservation(dobs, self.player),
                    threshold=self.options['threshold'],
                    rad=self.options['radius']
                )
            except KeyError:
                logging.error('options={}'.format(self.options))
                raise

            # gets overwritten intentionally, so we stick with the last
            self.planner = self.map_spawner.planner

            myteam = dobs['teams'][self.player]
            factories_to_place = myteam['factories_to_place']
            if self.max_allowed_factories is None:
                self.max_allowed_factories = factories_to_place
            self.monitor(step, dobs)

            my_turn_to_place = my_turn_to_place_factory(
                myteam['place_first'], step)

            # TODO: add logic to control metal and water allocation
            if factories_to_place > 0 and my_turn_to_place:
                spawn_loc = self.map_spawner.choose_spawn_loc()
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
        self.monitor(step, dobs)
        observation = CenteredObservation(dobs, self.player)
        self.update_oracle(observation)

        # Robot Building Logic
        self.build_robots()  # will update self.oracle['planned_actions']
        self.water()  # will update self.oracle['planned_actions']

        # # Robot moving logic
        # for unit_id in observation.my_units:
        #     robot_obs = RobotCenteredObservation(dobs, unit_id)
        #     robot_enacter = RobotEnacter(robot_obs, self.env_cfg)
        #     if robot_obs.queue_is_empty:
        #         ice_loc = self.get_ice(unit_id)
        #         if ice_loc:  # ice_loc is None if all ice already targeted
        #             actions[unit_id] = robot_enacter.ice_cycle(ice_loc)
        #             # TODO: add collision avoidance logic
        #         # TODO: think of else case here; at least move from plant

        return self.oracle['planned_actions']

    def monitor(self, step, dobs):
        o = CenteredObservation(dobs, self.player)
        assertion_msg = "step type{} step val{} len val{}".format(
            type(step), step, len(self.stats))
        assert step == (len(self.stats) + 1), assertion_msg

        # factories stats
        fc = deepcopy(o.total_factories_cargo)
        fc.update({
            'max_number_allowed': self.max_allowed_factories,
            'number_existing': len(o.my_factories),
            'power': np.sum(o.factories_ranked_by_power['power']),
        })

        # Robots stats
        rc = deepcopy(o.total_robots_cargo)

        def sum_robot_power(dict_iterator):
            return sum(v['power'] for v in dict_iterator)

        rc.update({
            'max_number_allowed': self.max_allowed_factories,
            'number_existing': len(o.my_units),
            'power': sum_robot_power(o.my_units.values()),
        })

        self.stats.append({'factories_total': fc, 'robots_total': rc})


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
        if environment.collect_stats:
            environment.state.stats[agent] = create_empty_stats()
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

    # iterate until phase 1 ends
    while env.state.real_env_steps < 0:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        if step == 1:
            first_obs = deepcopy(obs)
        imgs += [env.render("rgb_array", width=640, height=640)]
        if debug:
            logging.debug('{step}'.format(step=step))
    done = False

    inner_counter = 0
    while not done:
        if step >= steps:
            break
        actions = {}
        for player in env.agents:
            o = obs[player]
            if debug:
                a = agents[player].debug_act(step, o)
                logging.debug("{step}  {inner_counter}".format(
                    step=step, inner_counter=inner_counter))
            else:
                a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]
        inner_counter += 1
        if break_at_first_action and inner_counter == 2:
            break
    if animate_:
        logging.info('writing {animate_}'.format(animate_=animate_))
        return animate(imgs, filename=animate_)
    else:
        return first_obs, obs


if __name__ == "__main__":
    pass
