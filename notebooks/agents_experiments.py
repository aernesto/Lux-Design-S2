# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass, field
from luxai_s2.team import FactionTypes, Team
from luxai_s2.factory import Factory
from luxai_s2.state import State
from luxai_s2.env import EnvConfig, Board
from typing import Dict, Optional
from luxai_s2.unit import Unit
from collections import OrderedDict
import numpy as np
from luxai_s2.utils import animate, my_turn_to_place_factory

#
#  def obs_to_game_state(step, env_cfg: EnvConfig, obs):
#  units = dict()
#  for agent in obs["units"]:
#  units[agent] = dict()
#  for unit_id in obs["units"][agent]:
#  unit_data = obs["units"][agent][unit_id]
#  cargo = UnitCargo(**unit_data["cargo"])
#  unit = Unit(**unit_data,
#  unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
#  env_cfg=env_cfg)
#  unit.cargo = cargo
#  units[agent][unit_id] = unit
#
#  factory_occupancy_map = np.ones_like(obs["board"]["rubble"],
#  dtype=int) * -1
#  factories = dict()
#  for agent in obs["factories"]:
#  factories[agent] = dict()
#  for unit_id in obs["factories"][agent]:
#  f_data = obs["factories"][agent][unit_id]
#  cargo = UnitCargo(**f_data["cargo"])
#  factory = Factory(**f_data, env_cfg=env_cfg)
#  factory.cargo = cargo
#  factories[agent][unit_id] = factory
#  factory_occupancy_map[factory.pos_slice] = factory.strain_id
#  teams = dict()
#  for agent in obs["teams"]:
#  team_data = {k: obs['teams'][agent][k] for k in ['team_id', 'faction']}
#
#  faction = FactionTypes[team_data["faction"]]
#  teams[agent] = Team(**team_data, agent=agent)
#
#
#  NASTY                         GameMap(rubble=obs["board"]["rubble"],
#  ice=obs["board"]["ice"],
#  ore=obs["board"]["ore"],
#  lichen=obs["board"]["lichen"],
#  lichen_strains=obs["board"]["lichen_strains"],
#  factory_occupancy_map=factory_occupancy_map,
#  factories_per_team=obs["board"]["factories_per_team"],
#  valid_spawns_mask=obs["board"]["valid_spawns_mask"]),
#  return GameState(env_cfg=env_cfg,
#  env_steps=step,
#  board=Board(env_cfg=env_cfg, existing_map=map_),
#  units=units,
#  factories=factories,
#  teams=teams)
#
#
#  @dataclass
#  class GameState:
#  """
#  A GameState object at step env_steps. Copied from luxai_s2/state/state.py
#  """
#  env_steps: int
#  env_cfg: dict
#  board: Board
#  units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
#  factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
#  teams: Dict[str, Team] = field(default_factory=dict)
#
#  @property
#  def real_env_steps(self):
#  """
#  the actual env step in the environment, which subtracts the time spent bidding and placing factories
#  """
#  if self.env_cfg.BIDDING_SYSTEM:
#  return self.env_steps - (self.board.factories_per_team * 2 + 1)
#  else:
#  return self.env_steps
#
#  def is_day(self):
#  return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH
#


def reset_w_custom_board(environment, seed, custom_board):
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


class IdleAgent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.debug_act = self.act

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        actions = dict()
        try:
            factories_to_place = obs['teams'][
                self.player]['factories_to_place']
        except AttributeError:
            print(obs['teams'][self.player])
            raise
        my_turn_to_place = my_turn_to_place_factory(
            obs['teams'][self.player]['place_first'], step)

        if factories_to_place > 0 and my_turn_to_place:
            potential_spawns = np.array(
                list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
            #             logging.info(f'IdleAgent {potential_spawns.shape=}')
            selected_location_ix = -1
            spawn_loc = potential_spawns[selected_location_ix]
            #             logging.info(f'IdleAgent {spawn_loc=}')
            return dict(spawn=spawn_loc, metal=150, water=150)
        return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        return actions


#  def animate(imgs, _return=True):
#
#  video_name = ''.join(
#  random.choice(string.ascii_letters) for i in range(18)) + '.webm'
#  height, width, layers = imgs[0].shape
#  fourcc = cv2.VideoWriter_fourcc(*'VP90')
#  video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
#
#  for img in imgs:
#  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#  video.write(img)
#  video.release()
#  if _return:
#  from IPython.display import Video
#  return Video(video_name)
#


def interact(env,
             agents,
             steps,
             animate_: bool = True,
             break_at_first_action=False,
             debug=False,
             custom_board: Optional[Board] = None):
    # reset our env
    if custom_board is None:
        raise NotImplementedError()
    else:
        obs, env = reset_w_custom_board(env,
                                        seed=41,
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
        imgs += [env.render("rgb_array", width=640, height=640)]
        if debug:
            logging.info(f'{step=}')
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
                logging.info(f"{step=}  {inner_counter=}")
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
        return animate(imgs)
    else:
        return None


if __name__ == "__main__":
    pass
