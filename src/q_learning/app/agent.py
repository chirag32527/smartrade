import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory_size=10000, min_replay_memory_size=1000, 
                 gamma=0.99, learning_rate=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.min_replay_memory_size = min_replay_memory_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def replay(self):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        batch = random.sample(self.replay_memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            q_values = self.model.predict(state)
            q_target = reward
            if not done:
                q_target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            q_values[0][action] = q_target
            self.model.fit(state, q_values, verbose=0)

    def train(self, env, episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        rewards = []
        epsilon = epsilon_start
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_reward = 0
            while not done:
                action = self.act(state, epsilon)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()
            rewards.append(total_reward)
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay
            print("Episode: {}, total_reward: {}, epsilon: {:.4f}".format(episode+1, total_reward, epsilon))
        return rewards


env = OptionTradingEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

num_episodes = 1000
for e in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    i = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        i += 1
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, num_episodes, i, agent.epsilon))
