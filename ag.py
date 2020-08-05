from builtins import staticmethod
from copy import deepcopy

from tensorflow.keras.metrics import mse
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import clone_model

import random
import numpy as np


class Buffer:
    def __init__(self):
        self.s = None
        self.a = None
        self.sp = None
        self.r = None
        self.y = None


class Agent:
    LANDA = 0.9

    def __init__(self):
        self.qo = Agent.make_model()
        self.qt = Agent.make_model()

        self.buffer = [Buffer()]

    @staticmethod
    def make_model():
        model = Sequential()

        model.add(Conv2D(16, (4, 4), activation='relu', input_shape=(10, 10, 1)))
        model.add(Conv2D(16, (4, 4), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4))

        model.compile(loss="mean_squared_error", optimizer='adam', metrics=[mse])

        return model

    def predict(self, state):
        return self.qo.predict(state)

    def action(self, state):
        return np.argmax(self.predict(state))

    def add_buffer(self, state, a, reward):
        self.buffer[-1].sp = state
        new_buf = Buffer()
        new_buf.s = state
        new_buf.a = a
        new_buf.r = reward
        self.buffer.append(new_buf)
        return new_buf

        # fake_buff = Buffer()
        # fake_buff.s = Agent.make_fake_state(state, a)
        # fake_buff.r = 1
        # fake_buff.a = a
        # if fake_buff.s is not None:
        #     self.buffer.append(fake_buff)

    def make_batch(self, n=32):
        if len(self.buffer) <= n:
            return self.buffer[1:-1]
        return random.choices(self.buffer[1:-1], k=n)

    def train(self):
        batch = self.make_batch()
        if len(batch) == 0:
            return
        x_qt = [b.sp for b in batch]
        x_qo = [b.s for b in batch]
        y_t = self.qt.predict(np.asarray(x_qt))
        y_o = self.qo.predict(np.asarray(x_qo))

        for yt, yo, b in zip(y_t, y_o, batch):
            arg_max = np.argmax(yt)
            if b.r == -1 or b.r == 1:
                yo[b.a] = b.r
            else:
                yo[b.a] = b.r + Agent.LANDA * yt[arg_max]

        self.qo.fit(np.asarray(x_qo), y_o)

    def update_target(self):
        self.qt = clone_model(self.qo)
        self.qt.set_weights(self.qo.get_weights())

    @staticmethod
    def make_fake_state(state, a):
        pos = np.where(state == 1)
        pos = [pos[0][0], pos[1][0]]

        fake_state = deepcopy(state)
        if a == 0:
            pos[0] += 1
        if a == 2:
            pos[0] -= 1
        if a == 1:
            pos[1] -= 1
        if a == 3:
            pos[1] += 1

        if 0 <= pos[0] < 10 and 0 <= pos[1] < 10:
            fake_state[pos[0]][pos[1]] = 0.5
            return fake_state
        return None

    def make_fake_states(self, episode_buff):
        goal_pos = np.where(episode_buff[0].s == 0.5)
        goal_pos = [goal_pos[0][0], goal_pos[1][0]]
        new_goal_pos = np.where(episode_buff[-1].s == 1)
        new_goal_pos = [new_goal_pos[0][0], new_goal_pos[1][0]]

        for state in episode_buff[:-1]:
            new_state = deepcopy(state)
            # removing goal
            new_state.s[goal_pos[0]][goal_pos[1]][0] = 0
            new_state.sp[goal_pos[0]][goal_pos[1]][0] = 0

            if new_state.s[new_goal_pos[0]][new_goal_pos[1]][0] == 1:
                continue
            if new_state.sp[new_goal_pos[0]][new_goal_pos[1]][0] == 1:
                new_state.r = 1
                self.buffer = self.buffer[:-1] + [new_state, self.buffer[-1]]
                break
            # set goalpos
            new_state.s[new_goal_pos[0]][new_goal_pos[1]][0] = 0.5
            new_state.sp[new_goal_pos[0]][new_goal_pos[1]][0] = 0.5
            if state == episode_buff[-2]:
                new_state.r = 1
            else:
                new_state.r = Agent.set_reward(
                    np.where(new_state.s == 1),
                    np.where(new_state.sp == 1),
                    new_goal_pos
                )
            self.buffer = self.buffer[:-1] + [new_state, self.buffer[-1]]

    @staticmethod
    def set_reward(last_pos, new_pos, goal_pos):
        last_dist = (abs(last_pos[0][0] - goal_pos[0]) + abs(last_pos[1][0] - goal_pos[1]))
        new_dist = (abs(new_pos[0][0] - goal_pos[0]) + abs(new_pos[1][0] - goal_pos[1]))
        return (last_dist - new_dist) / 10
