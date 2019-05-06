from hlt_custom import commands
import random, time, os, sys, subprocess, logging, datetime, argparse, time
import numpy as np
from scipy.sparse import csr_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras.backend as K

# todo;
# error when saving too large numpy file, temp sol: restrict smaller replay capacity
# Bot error output was:
# 2019-05-03 17:33:19.042367: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# Using TensorFlow backend.
# Traceback (most recent call last):
#   File "rl_1.py", line 273, in <module>
#     np.save(os.path.join(folder, 'state_replay.npy'), state_replay)
#   File "/Users/Shared/anaconda/anaconda3/envs/hk01_py36/lib/python3.6/site-packages/numpy/lib/npyio.py", line 521, in save
#     pickle_kwargs=pickle_kwargs)
#   File "/Users/Shared/anaconda/anaconda3/envs/hk01_py36/lib/python3.6/site-packages/numpy/lib/format.py", line 593, in write_array
#     pickle.dump(array, fp, protocol=2, **pickle_kwargs)
# OSError: [Errno 22] Invalid argument


###########
# Network #
###########
class DQN(object):

    def __init__(self, input=(11, 32, 32), output=5):
        self.input = input
        self.output = output

        self.model = Sequential()
        self.model.add(Conv2D(24, (4, 4), strides=(2, 2), activation="relu", input_shape=self.input, data_format='channels_first'))
        self.model.add(Conv2D(48, (3, 3), strides=(1, 1), activation="relu", data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.output, activation=None))
        self.optimizer = keras.optimizers.RMSprop(lr=0.01, decay=0.0, epsilon=0.00001, rho=0.95)
        self.model.compile(loss=keras.losses.mean_squared_error, optimizer=self.optimizer)

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def fit(self, obs, act, rew, next_obs, done, discount=0.99, target_network=None, double=False):
        if not double:
            _q_value = self.model.predict(obs)  # 1d array
            _max_next_q_value = np.max(target_network.predict(next_obs), 1)  # 1d array
            y = act * (rew + (1 - done) * discount * _max_next_q_value)[:, None]  # 2d array
            y = _q_value * (1 - act) + (act * y)
        else:
            _q_value = self.model.predict(obs)  # 1d array
            _max_next_action = np.argmax(_q_value, 1)  # 1d array
            _max_next_double_q_value = target_network.predict(next_obs)[np.arange(len(obs)), _max_next_action] # 1d array
            y = act * (rew + (1 - done) * discount * _max_next_double_q_value)[:, None]  # 2d array
            y = _q_value * (1 - act) + (act * y)

        self.model.fit(obs, y, epochs=1, batch_size = len(obs), verbose=0)

    def save(self, path):
        self.model.save(path)

    def predict(self, x):
        return self.model.predict(x)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


###################
# Custom function #
###################
def preprocess(raw_state, me_id, turn_number, constants):
    """
    return processed state
    note the last one must be ship_id state
    note using sparse matrix could reduce >50% size of replay e.g. from 206mb to 62mb for 1100 records in state_replay.npy
    :param raw_state: dictionary of 2d ndarray
    :param me_id: int
    :return: 1d ndarray of dtype object, store nxn csr_matrix
    """
    halite_map = raw_state['halite_map']
    ship_halite_map = raw_state['ship_halite_map']
    ship_map = raw_state['ship_map']
    ship_id_map = raw_state['ship_id_map']
    shipyard_map = raw_state['shipyard_map']
    dropoff_map = raw_state['dropoff_map']

    try:
        constants.MAX_HALITE
    except:
        class _constants():
            def __init__(self):
                self.MAX_HALITE = 1000
                self.MOVE_COST_RATIO = 10
        constants = _constants()

    processed_state = []
    # one hot ship state
    processed_state.append(csr_matrix((ship_map == me_id) * 1.)) # 0
    processed_state.append(csr_matrix(((ship_map != me_id) & (ship_map != -1)) * 1.)) # 1
    # one hot shipyard state
    processed_state.append(csr_matrix((shipyard_map == me_id) * 1.)) # 2
    processed_state.append(csr_matrix(((shipyard_map != me_id) & (shipyard_map != -1)) * 1.)) # 3
    # one hot dropoff state
    # processed_state.append(csr_matrix((dropoff_map == me_id) * 1.))
    # processed_state.append(csr_matrix(((dropoff_map != me_id) & (dropoff_map != -1)) * 1.))
    # ship halite state
    processed_state.append(csr_matrix((ship_map == me_id) * ship_halite_map / constants.MAX_HALITE)) # 4
    processed_state.append(csr_matrix(processed_state[1].toarray() * ship_halite_map / constants.MAX_HALITE)) # 5
    # halite state
    processed_state.append(csr_matrix(halite_map / constants.MAX_HALITE)) # 6
    # moving cost state
    processed_state.append(csr_matrix(np.trunc(np.maximum(halite_map / constants.MOVE_COST_RATIO, 1) - ((halite_map == 0) * 1.)) / constants.MAX_HALITE)) # 7
    # remaining round
    processed_state.append(csr_matrix(np.ones(ship_map.shape) * (constants.MAX_TURNS - turn_number)))  # 8

    # todo add inspired halite / ship state / move cost

    # ship id state (this is used to center the matrix for different ship)
    processed_state.append(csr_matrix((-np.ones(ship_id_map.shape) * (ship_map != me_id)) + ship_id_map * (ship_map == me_id))) # This is not input to q_network

    return np.array(processed_state)


def center_state_for_ship(state, ship_id_map, ship_id, shape=None):
    """
    center the state from ship perspective
    :param state: 1d ndarray of csr matrix or 3d ndarray
    :param ship_id_map: 2d ndarray
    :param ship_id: int
    :param shape: None or int
    :return: 3d ndarray
    """

    _ship_location = np.where(ship_id_map == ship_id)
    _center_location = ((np.array(state[0].shape) - 1) / 2).astype(int)

    if len(_ship_location[0]) != 1:
        print('function center_state_for_ship: more then one / zero match found for ship_id {}, ship location {}'.format(ship_id, _ship_location), flush=True)
        return

    # convert into 3d ndarray if needed
    centered_state = state if len(state.shape) == 3 else np.array([_.toarray() for _ in state])

    for i in range(2):
        if _ship_location[i][0] == _center_location[0]: continue
        centered_state = np.roll(centered_state, _center_location[0] - _ship_location[i][0], i+1)

    # todo: crop state if shape < state[0].shape[0]

    return centered_state


def linearly_decaying_epsilon(step, warmup_steps, decay_period=250000, epsilon=0.01):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def get_halite_command(map_size=32, turn_limit=300, replay_directory='', parameters = {}):
    cmd = ["./halite", "--width %d" % map_size, "--height %d" % map_size, "--no-timeout", "--no-compression", "--seed 1557049871"]
    cmd.append("--turn-limit %d" % turn_limit)
    if replay_directory:
        cmd.append("--replay-directory %s" % (replay_directory))
        cmd.append("-vvv")
    cmd.append("python3 8_.py --MAX_SHIP_ON_MAP -1 --COLLISION_2P 0 --MAKE_DROPOFF_GAIN_COST_RATIO 999 --log_directory %s" % replay_directory)

    script = "python3 rl_1.py"
    for p in parameters.keys():
        script += ' --%s %s' % (p, parameters[p])
    script += ' --log_directory %s' % replay_directory

    cmd.append(script)
    return cmd


actions_list = np.array([commands.NORTH, commands.EAST, commands.SOUTH, commands.WEST, commands.STAY_STILL])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__name__)
    # General
    parser.add_argument("--max_training_steps", default=500000, type=int, help="Total number of training steps")
    parser.add_argument("--map_size", default=32, type=int, help="map size to be played")
    # Arg to be passed to rl_1.py
    parser.add_argument("--epsilon_train", default=0.01, type=float, help="the value to which the agent's epsilon is eventually decayed during training")
    parser.add_argument("--epsilon_decay_period", default=250000, type=int, help="length of the epsilon decay schedule")
    parser.add_argument("--batch_size", default=32, type=int, help="experiences fetched for training")
    parser.add_argument("--discount", default=0.99, type=float, help="the discount factor")
    parser.add_argument("--min_replay_history", default=20000, type=int, help="experiences needed before training q network")
    parser.add_argument("--target_update_period", default=8000, type=int, help="update period for the target network")
    parser.add_argument("--replay_capacity", default=50000, type=int, help="number of transitions to keep in memory")
    parser.add_argument("--folder", default='simple_dqn', type=str, help="folder to store networks, experiences, log")
    parser.add_argument("--double", default=0, type=int, help="use double q network or not")
    parser.add_argument("--sparse_reward", default=0, type=int, help="use sparse rewards")

    args = parser.parse_args()

    # parameter
    _parameters = {
        'epsilon_train': args.epsilon_train,
        'epsilon_decay_period': args.epsilon_decay_period,
        'batch_size': args.batch_size,
        'discount': args.discount,
        'min_replay_history': args.min_replay_history,
        'target_update_period': args.target_update_period,
        'training_steps': 0,
        'replay_capacity': args.replay_capacity,
        'folder': args.folder,
        'double': args.double,
        'sparse_reward': args.sparse_reward,
    }
    max_training_steps = args.max_training_steps
    map_size = args.map_size
    games_num = 0

    print(_parameters, flush=True)
    print('simple_dqn begin', flush=True)

    # Create folder
    if not os.path.exists(_parameters['folder']):
        os.mkdir(_parameters['folder'])

    # Set training_steps
    if os.path.exists(os.path.join(_parameters['folder'], 'ship_replay_index.npy')):
        ship_replay_index = np.load(os.path.join(_parameters['folder'], 'ship_replay_index.npy'))
        _parameters['training_steps'] = ship_replay_index
        del ship_replay_index

    # Init q_network and target_network
    if not os.path.exists(os.path.join(_parameters['folder'], 'q_network')):
        q_network = DQN(input=(9, map_size, map_size), output=len(actions_list))
        target_network = DQN(input=(9, map_size, map_size), output=len(actions_list))
        target_network.set_weights(q_network.get_weights())
        q_network.save(os.path.join(_parameters['folder'], 'q_network'))
        target_network.save(os.path.join(_parameters['folder'], 'target_network'))

    # Start experiment
    while _parameters['training_steps'] < max_training_steps:
        turn_limit = np.random.randint(2, 12) * 50 # 100, 150, ..., 550
        turn_limit = 400
        replay_directory = os.path.join(_parameters['folder'], datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(replay_directory):
            os.mkdir(replay_directory)
        cmd = get_halite_command(map_size=map_size, replay_directory = replay_directory, turn_limit = turn_limit + 2, parameters = _parameters)

        games_num += 1
        print('\nsimple_dqn starting %d game, trained %d steps\n' % (games_num, _parameters['training_steps']), flush=True)
        start_time = time.time()
        subprocess.call(cmd)
        # subprocess.call(["mv", "bot-0.log", "./{}/{}".format(replay_directory, "bot-0.log")])
        # subprocess.call(["mv", "bot-1.log", "./{}/{}".format(replay_directory, "bot-1.log")])
        print('\ntook {} seconds\n'.format(time.time() - start_time), flush=True)

        _parameters['training_steps'] += turn_limit

    print('simple_dqn done', flush=True)







