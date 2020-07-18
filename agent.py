import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from random import random, randint

import numpy as np

N_GAME_TRY = 10000
EPOCH_PER_STEP = 1

LANDA = 0.9
ALPHA = 0.9

env = gym.make('gym_pathfinder:PathFinder-v0')


def choose_act(dir, n_test):
    return dir
    p = N_GAME_TRY - n_test
    p /= N_GAME_TRY
    p -= 0.3
    if random() < p:
        return randint(0, 3)
    else:
        return dir


def game(model, n_test):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        print("###########################################")
        obs_f = np.ndarray((1, 100))
        obs_f[0] = obs.flatten()
        y = model.predict(obs_f)
        dir = np.argmax(y)
        print("y:", y)
        print("dir:", dir)

        action = choose_act(dir, n_test)
        obs, reward, done = env.step(action)
        env.render()

        if not done:
            new_obs_f = np.ndarray((1, 100))
            new_obs_f[0] = obs.flatten()
            new_y = model.predict(new_obs_f)
            new_dir = np.argmax(new_y)
            y[0][dir] = reward + LANDA * new_y[0][new_dir]
            print("ny:", y)
            model.fit(obs_f, y, epochs=EPOCH_PER_STEP)
        else:
            y[0][dir] = reward
            model.fit(obs_f, y, epochs=EPOCH_PER_STEP)

    env.close()


def game_dq(qa, qb, n_test):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        learner, other = (qa, qb) if random() < 0.5 else (qb, qa)
        print("###########################################")
        obs_f = np.ndarray((1, 100))
        obs_f[0] = obs.flatten()
        y_learner = learner.predict(obs_f)
        y_other = other.predict(obs_f)
        y = (y_learner + y_other) / 2
        dir = np.argmax(y)
        print("y:", y_learner)
        print("dir:", dir)

        action = choose_act(dir, n_test)
        obs, reward, done = env.step(action)
        env.render()
        print("r:", reward)
        if not done:
            new_obs_f = np.ndarray((1, 100))
            new_obs_f[0] = obs.flatten()
            new_y_learner = learner.predict(new_obs_f)
            new_y_other = other.predict(new_obs_f)
            new_dir = np.argmax(new_y_learner)
            y_learner[0][dir] = (y_learner[0][dir]
                                 + ALPHA
                                 * (reward + LANDA * new_y_other[0][new_dir]
                                    - y_learner[0][dir]))
            print("ny:", y_learner)
            learner.fit(obs_f, y_learner, epochs=EPOCH_PER_STEP)
        else:
            y_learner[0][dir] = reward
            y_other[0][dir] = reward
            learner.fit(obs_f, y_learner, epochs=EPOCH_PER_STEP)
            other.fit(obs_f, y_other, epochs=EPOCH_PER_STEP)

    env.close()


def make_model():
    model = Sequential()

    model.add(Dense(256, input_dim=100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4))

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

    return model


# r* landa * max(Q(s+1))
def main():
    qa = make_model()
    qb = make_model()
    for n_test in range(N_GAME_TRY):
        game_dq(qa, qb, n_test)


if __name__ == '__main__':
    main()
