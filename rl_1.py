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

# eval command
# ./halite --width 8 --height 8 --no-timeout --no-compression --seed 1557049871 --turn-limit 400 --replay-directory simple_dqn_8x8_3/eval/ -vvv "python3 8_.py --MAX_SHIP_ON_MAP -1 --COLLISION_2P 0 --MAKE_DROPOFF_GAIN_COST_RATIO 999 --log_directory simple_dqn_8x8_3/eval/" "python3 rl_1.py --eval 1 --folder simple_dqn_8x8_3/eval --log_directory simple_dqn_8x8_3/eval --epsilon_eval 0" > simple_dqn_8x8_3/eval/eval.log

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
parser.add_argument("--log_directory", default='', type=str, help="folder to store bot log")
parser.add_argument("--eval", default=0, type=int, help="evaluation mode, will not select random actions")
parser.add_argument("--epsilon_eval", default=0.01, type=float, help="epsilon used in evaluation mode")
# Network
parser.add_argument("--epsilon_train", default=0.01, type=float, help="the value to which the agent's epsilon is eventually decayed during training")
parser.add_argument("--epsilon_decay_period", default=250000, type=int, help="length of the epsilon decay schedule")
parser.add_argument("--batch_size", default=32, type=int, help="experiences fetched for training")
parser.add_argument("--discount", default=0.99, type=float, help="the discount factor")
parser.add_argument("--min_replay_history", default=20000, type=int, help="experiences needed before training q network")
parser.add_argument("--target_update_period", default=8000, type=int, help="update period for the target network")
parser.add_argument("--training_steps", default=0, type=int, help="number of steps trained (even just store experience) before")
parser.add_argument("--replay_capacity", default=10000, type=int, help="number of transitions to keep in memory")
parser.add_argument("--huber", default=1, type=int, help="use huber loss or not")
parser.add_argument("--double", default=1, type=int, help="use double q network or not")
parser.add_argument("--sparse_reward", default=0, type=int, help="use sparse rewards")

args = parser.parse_args()

if args.RANDOM_SEED != -1:
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)
folder = args.folder
log_directory = args.log_directory
eval = args.eval
epsilon_eval = args.epsilon_eval
epsilon_train = args.epsilon_train
epsilon_decay_period = args.epsilon_decay_period
batch_size = args.batch_size
discount = args.discount
min_replay_history = args.min_replay_history
target_update_period = args.target_update_period
training_steps = args.training_steps
replay_capacity = args.replay_capacity
huber = args.huber
double = args.double
sparse_reward = args.sparse_reward

start_time = time.time()

# Read network
q_network = DQN(huber=huber)
target_network = DQN(huber=huber)
q_network.load_model(os.path.join(folder, 'q_network'))
q_network.save(os.path.join(folder, 'q_network'))
target_network.load_model(os.path.join(folder, 'target_network'))

# Read replay
# todo: take care when the reaply capacity changed?
if os.path.exists(os.path.join(folder, 'state_replay.npy')) and os.path.exists(os.path.join(folder, 'ship_replay.npy')):
    state_replay = np.load(os.path.join(folder, 'state_replay.npy'))
    ship_replay = np.load(os.path.join(folder, 'ship_replay.npy'))
    state_replay_index = np.load(os.path.join(folder, 'state_replay_index.npy'))
    ship_replay_index = np.load(os.path.join(folder, 'ship_replay_index.npy'))
    _norm_state_replay_index = state_replay_index % state_replay.shape[0]
    _norm_ship_replay_index = ship_replay_index % ship_replay.shape[0]
else:
    state_replay = np.zeros((replay_capacity, 2, 10), dtype=object)
    ship_replay = np.zeros((replay_capacity, 5), dtype=object)
    state_replay_index = 0
    ship_replay_index = 0
    _norm_state_replay_index = 0
    _norm_ship_replay_index = 0
_episode_log = os.path.join(folder, 'episode_log.csv')

###################
# Custom function #
###################
def normalize_directional_offset(position, move):
    return game_map.normalize(position.directional_offset(move))

# def get_move_cost(position):
#     # get move cost of any position
#     if game_map[position].halite_amount == 0:
#         return 0
#     # elif game_map[position].halite_amount <= constants.MOVE_COST_RATIO:
#     #     return 1
#     else:
#         return int(game_map[position].halite_amount / constants.MOVE_COST_RATIO)


##################
# Main game loop #
##################
game = hlt.Game(log_directory)
game.ready("LouisBot")
logging.debug('{} seconds to read replay, network and ready game'.format(time.time() - start_time))

actions = {}
rewards = {}
terminations = {}
_info_ships_target_position = {}
_info_ships_move_cost = {}
_info_rewards_list = []
_info_loss_list = []

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
        for ship_id in rewards.keys():
            start_time = time.time()
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
                # todo: no ship object if crashed, need to find another way to calculate
                # if _move_to_dropoff:
                #     rewards[ship_id] += (ship.prev_halite_amount - _info_ships_move_cost[ship_id]) / constants.MAX_HALITE
            else:
                ship = me.get_ship(ship_id)
                if _move_to_dropoff:
                    rewards[ship_id] += (ship.prev_halite_amount - _info_ships_move_cost[ship_id]) / constants.MAX_HALITE
                else:
                    rewards[ship_id] += (ship.halite_amount - ship.prev_halite_amount) / constants.MAX_HALITE

            # override to use sparse rewards
            if sparse_reward and not _move_to_dropoff:
                rewards[ship_id] = 0

            _info_rewards_list.append(rewards[ship_id])

            logging.debug('{} seconds to calculate rewards / terminations'.format(time.time() - start_time))
            logging.debug('ship prev halite = {}, current halite = {}, move_cost = {}'.format(ship.prev_halite_amount, ship.halite_amount, _info_ships_move_cost[ship_id]))
        logging.debug('last round rewards = {}'.format(rewards))
        logging.debug('last round termination = {}'.format(terminations))

        # Append replay
        start_time = time.time()
        _ = np.array([[int(_norm_state_replay_index), id, actions[id], rewards[id], terminations[id]] for id in actions.keys()], dtype=object)[0]
        ship_replay[_norm_ship_replay_index] = _
        ship_replay_index += 1
        _norm_ship_replay_index = ship_replay_index % ship_replay.shape[0]
        _ = np.array([prev_processed_state, processed_state])
        state_replay[_norm_state_replay_index] = _
        state_replay_index += 1
        _norm_state_replay_index = state_replay_index % state_replay.shape[0]
        logging.debug('{} seconds to append replay'.format(time.time() - start_time))

        start_time = time.time()
        if not eval and ship_replay_index < batch_size:
            # Not enough sample to evaluate yet, just set loss to 0
            _info_loss_list.append(0)
        elif not eval:
            _samples_ship_index = np.random.choice(min(ship_replay.shape[0], ship_replay_index), batch_size, False)
            _samples_actions = np.array(list(ship_replay[_samples_ship_index, 2]), dtype=int)  # 2d array
            _samples_rewards = ship_replay[_samples_ship_index, 3].astype(np.float64) # 1d array
            _samples_termination = ship_replay[_samples_ship_index, 4].astype(int) # 1d array
            _samples_ship_id = ship_replay[_samples_ship_index, 1].astype(int) # 1d array
            _samples_state_index = ship_replay[_samples_ship_index, 0].astype(int)
            _samples_state = state_replay[_samples_state_index, 0]
            _samples_next_state = state_replay[_samples_state_index, 1]
            if len(_samples_state.shape) != 4:
                # transform 2d array of sparse matrix to 4d array
                _samples_state = np.array([c.toarray() for r in _samples_state for c in r ]).reshape(*_samples_state.shape, *_samples_state[0][0].shape)
                _samples_next_state = np.array([c.toarray() for r in _samples_next_state for c in r]).reshape(*_samples_next_state.shape, *_samples_next_state[0][0].shape)
            _samples_state_ship_id_map = _samples_state[:, -1, :, :]
            _samples_state = _samples_state[:, 0:-1, :, :] # 4d array of shape batch size x no. of features map x map_size x map_size, ignore ship id state
            _samples_next_state_ship_id_map = _samples_next_state[:, -1, :, :]
            _samples_next_state = _samples_next_state[:, 0:-1, :, :] # 4d array of shape batch size x no. of features map x map_size x map_size, ignore ship id state

            for i in range(batch_size):
                _samples_state[i] = center_state_for_ship(_samples_state[i], _samples_state_ship_id_map[i], _samples_ship_id[i])
                if _samples_termination[i]:
                    _samples_next_state[i] = center_state_for_ship(_samples_next_state[i], _samples_next_state_ship_id_map[i], _samples_ship_id[i])

            # Evaluate loss
            loss = q_network.evaluate(_samples_state, _samples_actions, _samples_rewards, _samples_next_state, _samples_termination, discount=discount, double=double, target_network=target_network)
            _info_loss_list.append(loss)

            if ship_replay_index > min_replay_history:
                # Fit the network
                q_network.fit(_samples_state, _samples_actions, _samples_rewards, _samples_next_state, _samples_termination, discount=discount, double=double, target_network=target_network)

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
    _move_cost_map = processed_state[7] * constants.MAX_HALITE

    # Set actions for every ships
    # todo: only do this if not reach MAX_TURNS?
    for ship in me.get_ships():
        start_time = time.time()
        centered_state = center_state_for_ship(processed_state[0:-1], processed_state[-1].toarray(), ship.id)

        # Select actions from epsilon-greedy policy
        _move_cost = _move_cost_map[ship.position.y, ship.position.x]
        epsilon = linearly_decaying_epsilon(training_steps, min_replay_history, epsilon_decay_period, epsilon_train) if not eval else epsilon_eval
        if ship.halite_amount < _move_cost and _move_cost > 0:
            a = commands.STAY_STILL
        elif np.random.rand() < epsilon:
            a = random.choice(actions_list)
            logging.info('epsilon={}, choosed random action={}'.format(epsilon, a))
        else:
            # _ = processed_state[0:-1] if len(processed_state.shape) == 3 else np.array([i.toarray() for i in processed_state[0:-1]])
            _q_val = q_network.predict(np.array([centered_state]))
            a = actions_list[np.argmax(_q_val)]
            logging.info('epsilon={}, q_val={}, a={}'.format(epsilon, _q_val, a))

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
        np.save(os.path.join(folder, 'state_replay_index.npy'), state_replay_index)
        np.save(os.path.join(folder, 'ship_replay_index.npy'), ship_replay_index)
        logging.debug('{} seconds to save replay'.format(time.time() - start_time))

        start_time = time.time()
        f = open(_episode_log, 'a')
        f.write('%f,%f,%f,%f,%f,%f,%f,%d\n'%(
            np.array(_info_rewards_list).mean(),
            np.array(_info_rewards_list).sum(),
            np.array(_info_rewards_list).max(),
            epsilon,
            np.array(_info_loss_list).mean(),
            np.array(_info_loss_list).sum(),
            np.array(_info_loss_list).max(),
            game.turn_number - 2
        ))
        f.close()
        logging.debug('{} seconds to write episode log'.format(time.time() - start_time))

        start_time = time.time()
        q_network.save(os.path.join(folder, 'q_network'))
        target_network.save(os.path.join(folder, 'target_network'))
        logging.debug('{} seconds to save network'.format(time.time() - start_time))

    # Spawn if no ship
    if len(me.get_ships()) == 0:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)


