from hlt import constants, commands
import random, time, os, sys, subprocess, logging
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras.backend as K


class DQN(object):

    def __init__(self, input=(84,84,4), output=5):
        self.input = input
        self.output = output

        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8), strides=(4, 4), activation="relu", input_shape=self.input))
        self.model.add(Conv2D(32, (4, 4), strides=(2, 2), activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.output, activation=None))
        self.optimizer = keras.optimizers.RMSprop(lr=0.00025, decay=0.95, epsilon=0.00001)
        self.model.compile(loss=keras.losses.mean_squared_error, optimizer=self.optimizer)


def preprocess(raw_state):
    processed_state = np.random.rand(84,84,4)
    return processed_state


def e_greedy(state, epsilon):
    actions = random.choice(actions_list) if random.random() <= epsilon else actions_list[np.argmax(q_network.model.predict(preprocess(state)))]
    return actions


def get_halite_command():
    cmd = ["./halite", "--width 32", "--height 32", "--no-timeout", "python3 8_.py --MAX_SHIP_ON_MAP 0"]
    script = "python3 rl_1.py"

    cmd.append(script)
    return cmd

episode = 100
T = 100
sync_target_network = 25
actions_list = np.array([commands.NORTH, commands.EAST, commands.SOUTH, commands.WEST, commands.STAY_STILL, commands.CONSTRUCT])



q_network = DQN(output=len(actions_list))
target_network = DQN(output=len(actions_list))
target_network.model.set_weights(q_network.model.get_weights())

q_network.model.save('q_network')
target_network.model.save('target_network')

for e in range(episode):

    # Sync target_network
    if (e + 1) % sync_target_network == 0:
        subprocess.call(["cp","-rp","q_network","target_network"])

    cmd = get_halite_command()
    subprocess.call(cmd)

logging.info('simple_dqn Done')







