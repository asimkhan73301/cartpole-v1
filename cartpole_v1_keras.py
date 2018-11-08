from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque, namedtuple
import numpy as np
import random
import gym


class DQN(object):
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.memory = deque(maxlen=1000)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_', 'done'))
        self.p_model = self._build_model()
        self.t_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24,  activation='relu'))
        model.add(Dense(24,  activation='relu'))
        model.add(Dense(self.action_size,  activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, *args):
        self.memory.append(self.Transition(*args))

    def choose_action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.p_model.predict(state)
        action_w_max_q = np.argmax(q_values[0])
        return action_w_max_q

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, state_, done = sample
            target = self.t_model.predict(state)

            if done:
                target[0][action] = reward
            else:
                q_ = max(self.t_model.predict(state_)[0])
                target[0][action] = reward + self.gamma * q_

            self.p_model.fit(state, target, epochs=1, verbose=0)


    def train_target_model(self):
        weights = self.p_model.get_weights()
        self.t_model.set_weights(weights)



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQN(env=env)

    scores = []
    EPISODES = 500
    TIMESTEPS = 500
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        for t in range(TIMESTEPS):
            action = agent.choose_action(state)

            state_, reward, done, _ = env.step(action)
            state_ = np.reshape(state_, [1, agent.state_size])
            agent.remember(state, action, reward, state_, done)

            state = state_
            total_reward += reward
            if done:
                scores.append(total_reward)
                print("EPISODE : {} / {}, SCORE: {}, REWARD: {} epsilon: {:.2} MEMORY LENGTH: {}".format(e, EPISODES, t, total_reward, agent.epsilon, len(agent.memory)))
                break

        agent.replay()
        agent.train_target_model()

    print("AVERAGE SCORE: {}".format(sum(scores) / len(scores)))