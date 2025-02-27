import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras


class OptionTradingNet(keras.Model):
    def __init__(self, num_actions):
        super(OptionTradingNet, self).__init__()
        self.input_layer = keras.layers.Input(shape=(5,))
        self.dense1 = keras.layers.Dense(128, activation="relu")
        self.dense2 = keras.layers.Dense(64, activation="relu")
        self.dense3 = keras.layers.Dense(32, activation="relu")
        self.logits = keras.layers.Dense(num_actions, activation="softmax")
        self.optimizer = keras.optimizers.Adam(lr=0.001)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.logits(x)
        return x


class OptionTradingEnv(gym.Env):
    def __init__(self):
        # observation space
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,))

        # action space
        self.action_space = gym.spaces.Discrete(6)

        # initial state
        self.difference_base_strike_price = 100
        self.premium_price = 5
        self.open_trades = []
        self.account_balance = 10000
        self.days_remaining_to_expiry = 30
        
        
        # Penalty params
        self.brokerage_penalty = 59
        self.expiry_stt_percentage_penalty = 0.125/100

        # define constants
        self.lot_size = 50

        # define neural network
        num_actions = 6
        input_shape = (5,)
        model = OptionTradingNet(num_actions=num_actions)
        model.build(input_shape)
        model.compile(optimizer=model.optimizer, loss="categorical_crossentropy")

    def step(self, action):
        # execute the action
        if action == 0:  # buy pe
            sell_order_found = False
            for trade in self.open_trades:
                if trade[0] == "pe" and trade[1] == self.difference_base_strike_price and trade[3] == "sell":
                    profit = (trade[2] - self.premium_price) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    sell_order_found = True
                    break
                
            if not sell_order_found:
                self.open_trades.append(("pe", self.difference_base_strike_price, self.premium_price, "buy"))
                
        elif action == 1:  # sell pe
            buy_order_found = False
            for trade in self.open_trades:
                if trade[0] == "pe" and trade[1] == self.difference_base_strike_price and trade[3] == "buy":
                    profit = (self.premium_price - trade[2]) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    buy_order_found = True
                    break
            if not buy_order_found:
                self.open_trades.append(("pe", self.difference_base_strike_price, self.premium_price, "sell"))
            
        elif action == 3:  # buy ce
            sell_order_found = False
            for trade in self.open_trades:
                if trade[0] == "ce" and trade[1] == self.difference_base_strike_price and trade[3] == "sell":
                    profit = (trade[2] - self.premium_price) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    sell_order_found = True
                    break
                
            if not sell_order_found:
                self.open_trades.append(("ce", self.difference_base_strike_price, self.premium_price, "buy"))
                
        elif action == 4:  # sell ce
            buy_order_found = False
            for trade in self.open_trades:
                if trade[0] == "ce" and trade[1] == self.difference_base_strike_price and trade[3] == "buy":
                    profit = (self.premium_price - trade[2]) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    buy_order_found = True
                    break
                
            if not buy_order_found:
                self.open_trades.append(("ce", self.difference_base_strike_price, self.premium_price, "sell"))
                
        elif action in [2, 5]:  # hold ce, pe
            pass

        # update the state
        self.days_remaining_to_expiry -= 1
        
        if self.days_remaining_to_expiry == 0 and len(self.open_trades) > 0:
            for trade in self.open_trades:
                if trade[3] == "buy":
                   profit = (self.premium_price - trade[2]) * self.lot_size
                else: # sell order
                    profit = (trade[2] - self.premium_price) * self.lot_size
                
                self.account_balance += profit
                self.account_balance -= self.brokerage_penalty
                stt_penalty_amount = profit*self.expiry_stt_percentage_penalty
                # 18% gst on stt
                self.account_balance -= (1.18 * stt_penalty_amount)

        # calculate the reward
        reward = self.account_balance

        # check if the episode is done
        done = self.days_remaining_to_expiry == 0

        # return the next observation, reward, done flag, and any additional information
        return self._get_obs(), reward, done, {}

    def reset(self):
        # reset the state
        self.difference_base_strike_price = 100
        self.premium_price = 5
        self.open_trades = []
        self.account_balance = 10000
        self.days_remaining_to_expiry = 30

        return self._get_obs()

    def render(self, mode="human"):
        # print the current state
        print("Base Price: {}".format(self.difference_base_strike_price))
        print("Premium Price: {}".format(self.premium_price))
        print("Open Trades: {}".format(self.open_trades))
        print("Account Balance: {}".format(self.account_balance))
        print("Days Remaining to Expiry: {}".format(self.days_remaining_to_expiry))

    def _get_obs(self):
        # return the current observation
        return np.array(
            [
                self.difference_base_strike_price,
                self.premium_price,
                len(self.open_trades),
                self.account_balance,
                self.days_remaining_to_expiry,
            ]
        )
