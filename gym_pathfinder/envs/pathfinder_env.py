import gym
from gym import spaces
import numpy as np
import random
import os


class PathFinderEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PathFinderEnv, self).__init__()
        self.sparse_reward = False
        self.count_i = 10
        self.count_j = 10
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([3]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.count_i, self.count_j), dtype=np.float16)
        self.current_step = 0
        self.current_map = None
        self.viewer = None
        self.state = None
        self.agent_position = None
        self.goal_position = None
        self.walls = None
        self.wall_value = 0.25
        self.free_value = 0.0
        self.goal_value = 0.5
        self.agent_value = 1.0
        self.random_walls = 5
        self.map_path = {}
        self.maps = {}
        self.rot_maps = []
        self.diagonal_maps = []
        self.use_dia_map = False
        self.time_neg_reward = False
        self.move_neg_reward = False

    def add_map_path(self, name, path, rot_name=None):
        self.map_path[name] = path
        self.maps[name] = []
        if rot_name is not None:
            self.maps[rot_name] = []
        for map_file in os.listdir(path):
            map_path = os.path.join(path, map_file)
            f = open(map_path, 'r').readline()
            vmap = eval(f)
            self.maps[name].append(vmap)
            if rot_name is not None:
                self.maps[rot_name].append([[w[1], w[0]] for w in vmap])

    def _next_observation(self, done):
        obs = np.zeros(self.observation_space.shape)
        if not done:
            obs[self.agent_position[0], self.agent_position[1]] = self.agent_value
            obs[self.goal_position[0], self.goal_position[1]] = self.goal_value
            for w in self.walls:
                obs[w[0], w[1]] = self.wall_value
        obs = obs.reshape((10, 10, 1))
        return obs

    def _take_action(self, action):
        if action == 0:
            self.agent_position[0] += 1
        if action == 2:
            self.agent_position[0] -= 1
        if action == 1:
            self.agent_position[1] -= 1
        if action == 3:
            self.agent_position[1] += 1

    def reward_calculator(self, state, next_state):
        goal_position = np.where(state == self.goal_value)
        agent_position = np.where(state == self.agent_value)
        next_agent_position = np.where(next_state == self.agent_value)
        done = False
        if self.sparse_reward:
            if len(next_agent_position[0]) == 0:
                reward = 2
                done = True
            elif goal_position == next_agent_position:
                reward = 2
                done = True
            else:
                reward = -2
        else:
            if len(next_agent_position[0]) == 0:
                reward = 2
                done = True
            elif goal_position == next_agent_position:
                reward = 2
                done = True
            elif next_agent_position[0] < 0 \
                    or next_agent_position[0] > self.observation_space.shape[0] \
                    or next_agent_position[1] < 0 or next_agent_position[1] > self.observation_space.shape[1] \
                    or next_agent_position in self.walls:
                reward = -2
                done = True
            else:
                c_diff = abs(next_agent_position[0] - goal_position[0]) + abs(
                    next_agent_position[1] - goal_position[1])
                p_diff = abs(agent_position[0] - goal_position[0]) + abs(
                    agent_position[1] - goal_position[1])
                reward = (p_diff - c_diff) / 20
                if self.move_neg_reward:
                    reward -= 0.02
                if self.time_neg_reward:
                    if self.current_step > 50:
                        reward = -2
        return reward, done

    def step(self, action):
        previous_position = [self.agent_position[0], self.agent_position[1]]
        self._take_action(action)
        self.current_step += 1
        reward = 0
        if self.sparse_reward:
            if self.goal_position == self.agent_position:
                reward = 2
            else:
                reward = -2
        else:
            if self.goal_position == self.agent_position:
                reward = 2
            elif self.agent_position[0] < 0 \
                    or self.agent_position[0] > self.observation_space.shape[0] \
                    or self.agent_position[1] < 0 or self.agent_position[1] > self.observation_space.shape[1] \
                    or self.agent_position in self.walls:
                reward = -2
            else:
                c_diff = abs(self.agent_position[0] - self.goal_position[0]) + abs(
                    self.agent_position[1] - self.goal_position[1])
                p_diff = abs(previous_position[0] - self.goal_position[0]) + abs(
                    previous_position[1] - self.goal_position[1])
                reward = (p_diff - c_diff) / 50

                if self.move_neg_reward:
                    reward -= 0.02
                if self.time_neg_reward:
                    if self.current_step > 50:
                        reward = -2

        done = bool(self.goal_position == self.agent_position
                    or self.agent_position[0] < 0
                    or self.agent_position[0] > self.observation_space.shape[0] - 1
                    or self.agent_position[1] < 0
                    or self.agent_position[1] > self.observation_space.shape[1] - 1
                    or self.agent_position in self.walls
                    or self.current_step > 50
                    )
        information = {'result': 'normal'}
        if done:
            if self.goal_position == self.agent_position:
                information['result'] = 'goal'
            elif self.agent_position in self.walls:
                information['result'] = 'wall'
            elif self.current_step > 50:
                information['result'] = 'time'
            else:
                information['result'] = 'out'

        obs = self._next_observation(done)
        self.state = obs
        return obs, reward, done, information

    def agent_to_goal_available(self):
        map = [[0 for _ in range(10)] for _ in range(10)]
        for w in self.walls:
            map[w[0]][w[1]] = -1
        map[self.agent_position[0]][self.agent_position[1]] = 1
        map[self.goal_position[0]][self.goal_position[1]] = 2
        queue = [[self.agent_position[0], self.agent_position[1]]]
        while len(queue) > 0:
            center = queue[0]
            neiburs = [[center[0] + 1, center[1]], [center[0] - 1, center[1]], [center[0], center[1] - 1],
                       [center[0], center[1] + 1]]
            for n in neiburs:
                if n[0] >= 10 or n[0] < 0 or n[1] >= 10 or n[1] < 0:
                    continue
                if map[n[0]][n[1]] == 0:
                    queue.append([n[0], n[1]])
                    map[n[0]][n[1]] = 1
                if map[n[0]][n[1]] == 2:
                    return True
            del queue[0]
        return False

    def reset(self, map_name=None):
        wall_inserted = 0
        free_cell = [[i, j] for i in range(10) for j in range(10)]
        self.walls = []
        self.current_map = []
        if map_name is not None:
            self.current_map = random.choice(self.maps[map_name])
        for w in self.current_map:
            self.walls.append(w)
            free_cell.remove(w)
        random.shuffle(free_cell)
        while wall_inserted < self.random_walls:
            wall = free_cell[0]
            self.walls.append(wall)
            free_cell.remove(wall)
            wall_inserted += 1

        while True:
            random.shuffle(free_cell)
            self.agent_position = free_cell[0]
            self.goal_position = free_cell[1]
            if self.agent_to_goal_available():
                break
        self.current_step = 0
        self.state = np.zeros(self.observation_space.shape)
        obs = self._next_observation(False)
        self.state = obs
        return obs

    def render(self, mode='human', close=False):
        screen_width = 400
        screen_height = 400
        cell_width = screen_width / self.observation_space.shape[0]
        cell_height = screen_height / self.observation_space.shape[1]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.cells = [[None for i in range(self.observation_space.shape[0])] for j in
                          range(self.observation_space.shape[1])]
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(self.observation_space.shape[1]):
                for j in range(self.observation_space.shape[0]):
                    l, r, b, t = j * cell_width, \
                                 (j + 1) * cell_width - 1, \
                                 (self.observation_space.shape[0] - i - 1) * cell_height, \
                                 (self.observation_space.shape[0] - i) * cell_height - 1
                    cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    cell.set_color(0.5, 0.5, 0.5)
                    self.cells[i][j] = cell
                    self.viewer.add_geom(cell)

        for i in range(self.observation_space.shape[0]):
            for j in range(self.observation_space.shape[1]):
                cell = self.cells[i][j]
                if self.agent_position == [i, j]:
                    cell.set_color(0, 0, 1)
                elif self.goal_position == [i, j]:
                    cell.set_color(0, 1, 0)
                elif [i, j] in self.walls:
                    cell.set_color(0., 0., 0.)
                else:
                    cell.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(cell)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
