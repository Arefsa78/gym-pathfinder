from keras import models, layers, activations, optimizers, losses, metrics, regularizers
from keras.engine.sequential import Sequential
import random
from numpy import array
from typing import List
import sys
import copy
import keras
from keras import Model
from keras.layers.merge import concatenate
import numpy as np
from agents.reply_buffer_simple import Buffer
from agents.transit import Transition


class DeepQ:
    def __init__(self, buffer_size=100000, train_interval_step=100, target_update_interval_step=200, prb=False, train_step_counter=3200):
        self.model_type = ''
        self.model = None
        self.target_network = None
        self.prb = prb
        self.buffer = Buffer(buffer_size)
        self.train_interval_step = train_interval_step
        self.target_update_interval_step = target_update_interval_step
        self.train_step_counter = train_step_counter
        self.action_number = 4
        self.transitions: List[Transition] = []
        self.gama = 0.99
        self.episode_number = 0
        self.plan_number = 0
        self.step_number = 0
        self.use_double = False
        self.loss_values = []
        self.rotating = False
        pass

    def create_model_cnn_dense(self):
        self.model_type = 'image'
        in1 = layers.Input((10, 10, 1,))
        m1 = layers.Conv2D(16, (2, 2), strides=(1, 1), activation='relu')(in1)
        m1 = layers.Conv2D(16, (2, 2), strides=(1, 1), activation='relu')(m1)
        m1 = layers.Conv2D(16, (2, 2), strides=(1, 1), activation='relu')(m1)
        m1 = layers.Flatten()(m1)
        out = layers.Dense(512, activation='relu')(m1)
        out = layers.Dense(256, activation='relu')(out)
        out = layers.Dense(64, activation='relu')(out)
        out = layers.Dense(self.action_number)(out)

        model = keras.Model(in1, out)
        model.compile(optimizer=optimizers.Adam(lr=0.00025), loss=losses.mse, metrics=[metrics.mse])
        model.summary()
        self.model = model
        self.target_network = keras.models.clone_model(self.model)

    def create_model_dense(self, input_shape=(4,)):
        self.model_type = 'param'
        model = models.Sequential()
        model.add(layers.Dense(100, activation=activations.relu, input_shape=input_shape))
        model.add(layers.Dense(60, activation=activations.relu))
        model.add(layers.Dense(30, activation=activations.relu))
        model.add(layers.Dense(9, activation=activations.linear))
        model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        model.summary()
        self.model = model
        self.target_network = keras.models.clone_model(self.model)

    def read_model(self, path, model_type):
        self.model_type = model_type
        self.model = keras.models.load_model(path)
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95), loss=losses.mse, metrics=[metrics.mse])
        self.target_network = keras.models.clone_model(self.model)
        print(self.model.summary())

    def read_weight(self, path):
        self.model.load_weights(path)
        self.target_network = keras.models.clone_model(self.model)

    def get_q(self, transits: List[Transition]):
        x = []
        for t in transits:
            x.append(t.state)
        x = array(x)
        y = self.model.predict(x)
        return y

    def get_best_action(self, state):
        if self.model_type == 'image':
            if state.ndim == 3:
                state = state.reshape((1, 10, 10, 1))
        elif self.model_type == 'param':
            if state.ndim == 1:
                state = state.reshape((1, state.shape[0]))

        Y = self.model.predict(state)
        actions = np.argmax(Y, axis=1)
        max_qs = np.max(Y, axis=1).flatten()

        return actions, max_qs

    def get_random_action(self, state, p_rnd=0.1):
        if random.random() < p_rnd:
            return random.randrange(self.action_number)
        best_action, best_q = self.get_best_action(state)
        return best_action[0]

    @staticmethod
    def rotate_action(ac):
        action_rot = [3, 0, 1, 2]
        return action_rot[ac]

    def add_to_buffer(self, state, action, reward, next_state, done, train=True):
        if done:
            next_state = None

        transition = Transition(state, action, reward, next_state)
        self.buffer.add(transition)
        if self.rotating:
            for i in range(3):
                state = np.rot90(state)
                if next_state is not None:
                    next_state = np.rot90(next_state)
                action = DeepQ.rotate_action(action)
                transition = Transition(state, action, reward, next_state)
                self.buffer.add(transition)
        if train:
            self.train()
            if next_state is None:  # End step in episode
                self.episode_number += 1

    def train(self):
        self.step_number += 1
        if self.step_number % self.train_interval_step == 0:
            self.update_from_buffer()
        if self.step_number % self.target_update_interval_step == 0:
            self.target_network.set_weights(self.model.get_weights())

    def update_from_buffer(self):
        transits: List[Transition] = self.buffer.get_rand(self.train_step_counter)
        if len(transits) == 0:
            return
        is_image_param = False
        if self.model_type == 'imageparam':
            is_image_param = True

        states_view = []
        next_states_view = []
        states_param = []
        next_states_param = []
        for t in transits:
            if is_image_param:
                states_view.append(t.state[0])
                states_param.append(t.state[1])
                if t.is_end:
                    next_states_view.append(t.state[0])
                    next_states_param.append(t.state[1])
                else:
                    next_states_view.append(t.next_state[0])
                    next_states_param.append(t.next_state[1])
            else:
                states_view.append(t.state)
                if t.is_end:
                    next_states_view.append(t.state)
                else:
                    next_states_view.append(t.next_state)

        if is_image_param:
            states_view = array(states_view)
            next_states_view = array(next_states_view)
            states_param = array(states_param)
            next_states_param = array(next_states_param)
        else:
            states_view = array(states_view)
            next_states_view = array(next_states_view)

        if is_image_param:
            states_view = [states_view, states_param]
            next_states_view = [next_states_view, next_states_param]
        q = self.model.predict(states_view)
        best_q_action = np.argmax(q, axis=1)
        next_q = self.target_network.predict(next_states_view)

        if self.use_double:
            next_states_max_q = []
            for i in best_q_action:
                next_states_max_q.append(next_q[len(next_states_max_q)][i])
            next_states_max_q = np.array(next_states_max_q)
        else:
            next_states_max_q = np.max(next_q, axis=1).flatten()

        for i in range(len(transits)):
            q_learning = transits[i].reward
            if not transits[i].is_end:
                q_learning += (self.gama * next_states_max_q[i])
            diff = (q_learning - q[i][transits[i].action]) * transits[i].value
            q[i][transits[i].action] += diff

        history = self.model.fit(states_view, q,  epochs=1, batch_size=32, verbose=0)
        history_dict = history.history
        loss_values = history_dict['loss']
        self.loss_values.append(loss_values[0])


if __name__ == '__main__':
    rl = DeepQ()
    rl.create_model_cnn_dense()
