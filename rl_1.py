#!/usr/bin/env python3
# equivalent to bot version 13 & 15

#####################################################################################################################
#
# Change History
#
# Date                  Desc
# ----                  ----
# 2019-03-13 00:00      Initial version
#
#####################################################################################################################


# Todo
# 1.

#########
# Setup #
#########
import hlt_custom as hlt
from hlt_custom import constants, commands
import random
import logging
import argparse, time, numpy as np
from simple_dqn import *
import keras

parser = argparse.ArgumentParser(__name__)
# General
parser.add_argument("--RANDOM_SEED", default=-1, type=int, help="random seed, -1 = no need to set the seed")
parser.add_argument("--folder", default='', type=str, help="folder to store networks, experiences, log")
# Network
parser.add_argument("--epsilon_train", default=0.01, type=float, help="the value to which the agent's epsilon is eventually decayed during training")
parser.add_argument("--epsilon_decay_period", default=250000, type=int, help="length of the epsilon decay schedule")
parser.add_argument("--batch_size", default=32, type=int, help="experiences fetched for training")
parser.add_argument("--discount", default=0.99, type=float, help="the discount factor")
parser.add_argument("--min_replay_history", default=20000, type=int, help="experiences needed before training q network")
parser.add_argument("--target_update_period", default=8000, type=int, help="update period for the target network")
parser.add_argument("--training_steps", default=0, type=int, help="number of steps trained (even just store experience) before")

args = parser.parse_args()

if args.RANDOM_SEED != -1:
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
folder = args.folder
epsilon_train = args.epsilon_train
epsilon_decay_period = args.epsilon_decay_period
batch_size = args.batch_size
discount = args.discount
min_replay_history = args.min_replay_history
target_update_period = args.target_update_period
training_steps = args.training_steps

start_time = time.time()

# Read network
q_network = keras.models.load_model(os.path.join(folder, 'q_network'))
target_network = keras.models.load_model(os.path.join(folder, 'target_network'))

# Read replay
if os.path.exists(os.path.join(folder, 'state_replay.npy')) and os.path.exists(os.path.join(folder, 'ship_replay.npy')):
    state_replay = np.load(os.path.join(folder, 'state_replay.npy'))
    ship_replay = np.load(os.path.join(folder, 'ship_replay.npy'))
else:
    state_replay = np.array([])
    ship_replay = np.array([])


###################
# Custom function #
###################
def normalize_directional_offset(position, move):
    return game_map.normalize(position.directional_offset(move))

def get_move_cost(position):
    # get move cost of any position
    if game_map[position].halite_amount == 0:
        return 0
    elif game_map[position].halite_amount <= constants.MOVE_COST_RATIO:
        return 1
    else:
        return int(game_map[position].halite_amount / constants.MOVE_COST_RATIO)


##################
# Main game loop #
##################
game = hlt.Game()
game.ready("LouisBot")
logging.debug('{} seconds to read replay, network and ready game'.format(time.time() - start_time))

actions = {}
rewards = {}
terminations = {}
_info_ships_target_position = {}
_info_ships_move_cost = {}

while True:
    """
    Note game.turn_number starts from 1 to constants.MAX_TURNS

    While True:
        if actions set in last turn:
            train q network 

        if any ship:
            set actions
        else:
            spawn ship

        End
    """


    # Get the latest game state.
    # Extract player metadata and the updated map metadata convenience.
    start_time = time.time()
    game.update_frame()
    me = game.me
    game_map = game.game_map
    command_queue = []
    logging.debug('{} seconds to init game state'.format(time.time() - start_time))

    # Set raw_state and processed_state
    raw_state = {
        'halite_map': game_map.halite_map,
        'ship_halite_map': game_map.ship_halite_map,
        'ship_map': game_map.ship_map,
        'ship_id_map': game_map.ship_id_map,
        'shipyard_map': game_map.shipyard_map,
        'dropoff_map': game_map.dropoff_map,
    }
    processed_state = preprocess(raw_state, me.id, game.turn_number, constants)

    # only train if there is any action last round
    if len(actions.keys()) != 0:
        # Calculate rewards, terminations
        # todo: calculate rewards if converted into dropoff
        start_time = time.time()
        for ship_id in rewards.keys():
            # Check if ship moved to dropoff
            _move_to_dropoff = False
            if ship_id in _info_ships_target_position.keys():
                _target_position = _info_ships_target_position[ship_id]
                if raw_state['shipyard_map'][_target_position] == 1 or raw_state['dropoff_map'][_target_position] == 1:
                    _move_to_dropoff = True

            if game.turn_number == constants.MAX_TURNS:
                terminations[ship_id] = 1

            if not me.has_ship(ship_id):
                # If ship terminated
                terminations[ship_id] = 1
                rewards[ship_id] += -1
                # todo: no ship object if crashed
                # if _move_to_dropoff:
                #     rewards[ship_id] += (ship.prev_halite_amount - _info_ships_move_cost[ship_id]) / constants.MAX_HALITE
            else:
                ship = me.get_ship(ship_id)
                if _move_to_dropoff:
                    rewards[ship_id] += (ship.prev_halite_amount - _info_ships_move_cost[ship_id]) / constants.MAX_HALITE
                else:
                    rewards[ship_id] += (ship.halite_amount - ship.prev_halite_amount) / constants.MAX_HALITE
        logging.debug('{} seconds to calculate rewards / terminations'.format(time.time() - start_time))

        # Append replay
        start_time = time.time()
        _ = np.array([[prev_processed_state, processed_state]])
        state_replay = _ if len(state_replay) == 0 else np.append(state_replay, _, axis=0)
        _ = np.array([[state_replay.shape[0] - 1, id, actions[id], rewards[id], terminations[id]] for id in actions.keys()], dtype=object)
        ship_replay = _ if len(ship_replay) == 0 else np.append(ship_replay, _, axis=0)
        logging.debug('{} seconds to append replay'.format(time.time() - start_time))

        # Fit the network
        start_time = time.time()
        if len(state_replay) > min_replay_history:
            _samples_ship_index = np.random.choice(ship_replay.shape[0], batch_size, False)
            _samples_actions = np.array(list(ship_replay[_samples_ship_index, 2]), dtype=int)  # 2d array
            _samples_rewards = ship_replay[_samples_ship_index, 3].astype(np.float64) # 1d array
            _samples_termination = ship_replay[_samples_ship_index, 4].astype(int) # 1d array
            _samples_state_index = ship_replay[_samples_ship_index, 0].astype(int)
            _samples_state = state_replay[_samples_state_index, 0]
            _samples_next_state = state_replay[_samples_state_index, 1]
            if len(_samples_state.shape) != 4:
                # transform 2d array of sparse matrix to 4d array
                _samples_state = np.array([c.toarray() for r in _samples_state for c in r ]).reshape(*_samples_state.shape, *_samples_state[0][0].shape)
                _samples_next_state = np.array([c.toarray() for r in _samples_next_state for c in r]).reshape(*_samples_next_state.shape, *_samples_next_state[0][0].shape)
            _samples_state = _samples_state[:, 0:-1, :, :] # 4d array of shape batch sizex11x32x32, ignore ship id state
            _samples_next_state = _samples_next_state[:, 0:-1, :, :] # 4d array of shape batch sizex11x32x32, ignore ship id state

            _q_value = q_network.predict(_samples_state) # 1d array
            _max_next_q_value = np.max(q_network.predict(_samples_next_state), 1) # 1d array
            y = _samples_actions * (_samples_rewards + (1 - _samples_termination) * discount * _max_next_q_value)[:, None] # 2d array
            y = _q_value * (1 - _samples_actions) + (_samples_actions * y)
            x = _samples_state

            q_network.fit(x, y, epochs=1, verbose=0)

            # Update target network
            if training_steps % target_update_period == 0:
                target_network.set_weights(q_network.get_weights())
        logging.debug('{} seconds to fit the nerwork'.format(time.time() - start_time))

        training_steps += 1

    # Reset
    prev_raw_state = raw_state
    prev_processed_state = processed_state
    actions = {}
    rewards = {}
    terminations = {}
    _info_ships_target_position = {}
    _info_ships_move_cost = {}
    _move_cost_map = processed_state[9] * constants.MAX_HALITE

    # Set actions for every ships
    # todo: only do this if not reach MAX_TURNS?
    for ship in me.get_ships():
        start_time = time.time()
        state = center_state_for_ship(processed_state[0:-1], processed_state[-1].toarray(), ship.id)

        # Select actions from epsilon-greedy policy
        _move_cost = _move_cost_map[ship.position.y, ship.position.x]
        epsilon = linearly_decaying_epsilon(training_steps, min_replay_history, epsilon_decay_period, epsilon_train)
        if ship.halite_amount < _move_cost:
            a = commands.STAY_STILL
        elif np.random.rand() < epsilon:
            a = random.choice(actions_list)
        else:
            _ = processed_state[0:-1] if len(processed_state.shape) == 3 else np.array([i.toarray() for i in processed_state[0:-1]])
            a = actions_list[np.argmax(q_network.predict(np.array([_])))]

        if a in [commands.NORTH, commands.EAST, commands.SOUTH, commands.WEST, commands.STAY_STILL]:
            target_position = normalize_directional_offset(ship.position, a)
            _info_ships_target_position[ship.id] = (target_position.y, target_position.x)
            _info_ships_move_cost[ship.id] = _move_cost
        actions[ship.id] = np.array(actions_list == a) * 1.
        terminations[ship.id] = 0
        rewards[ship.id] = 0

        # Append command to queue
        if a in [commands.NORTH, commands.EAST, commands.SOUTH, commands.WEST, commands.STAY_STILL]:
            command = ship.move(a)
        elif a == commands.CONSTRUCT:
            if __name__ == '__main__':
                command = ship.make_dropoff()
        command_queue.append(command)
        logging.debug('{} seconds to set acions for one ship'.format(time.time() - start_time))

    if game.turn_number == constants.MAX_TURNS:
        start_time = time.time()
        np.save(os.path.join(folder, 'state_replay.npy'), state_replay)
        np.save(os.path.join(folder, 'ship_replay.npy'), ship_replay)
        logging.debug('{} seconds to save replay'.format(time.time() - start_time))

        start_time = time.time()
        q_network.save(os.path.join(folder, 'q_network'))
        target_network.save(os.path.join(folder, 'target_network'))
        logging.debug('{} seconds to save network'.format(time.time() - start_time))

    # Spawn if no ship
    if len(me.get_ships()) == 0:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)





