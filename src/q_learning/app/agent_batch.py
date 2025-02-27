import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory_size=5000, min_replay_memory_size=1000,
                 gamma=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = []
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)
            
    def replay(self, batch_size):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        batch = np.array(random.sample(self.replay_memory, batch_size))
        states = batch[:, 0]
        actions = batch[:, 1]
        rewards = batch[:, 2]
        next_states = batch[:, 3]
        dones = batch[:, 4]
        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        targets_full[np.arange(batch_size), actions] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, env, episodes, batch_size):
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay(batch_size)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
