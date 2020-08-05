import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from random import random, randint

import numpy as np

from ag import Agent

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


def game_hcl():
    agent = Agent()

    for epoch in range(1000):
        # train
        for n_train in range(200):
            print(f"********{epoch}-{n_train}********")
            obs = env.reset()
            env.render()
            done = False
            episode_buff = []
            while not done:
                agent.train()

                n_obs = obs.reshape(1, 10, 10, 1)
                action = agent.action(n_obs)
                obs, reward, done = env.step(action)

                if not done:
                    n_obs = obs.reshape(10, 10, 1)
                    step_buff = agent.add_buffer(n_obs, action, reward)
                    episode_buff.append(step_buff)

                env.render()
            if len(episode_buff) > 3 and False:
                agent.make_fake_states(episode_buff)
            env.close()

        # test
        for n_test in range(10):
            pass

        agent.update_target()


# r* landa * max(Q(s+1))
def main():
    game_hcl()


if __name__ == '__main__':
    main()
