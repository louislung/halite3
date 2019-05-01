from hlt_custom import commands
import random, time, os, sys, subprocess, logging, datetime
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras.backend as K

###########
# Network #
###########
class DQN(object):

    def __init__(self, input=(12, 32, 32), output=5):
        self.input = input
        self.output = output

        self.model = Sequential()
        self.model.add(Conv2D(24, (4, 4), strides=(2, 2), activation="relu", input_shape=self.input))
        self.model.add(Conv2D(48, (3, 3), strides=(1, 1), activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.output, activation=None))
        self.optimizer = keras.optimizers.RMSprop(lr=0.00025, decay=0.95, epsilon=0.00001)
        self.model.compile(loss=keras.losses.mean_squared_error, optimizer=self.optimizer)


###################
# Custom function #
###################
def preprocess(raw_state, me_id, turn_number, constants):
    """
    return processed state
    note the last one must be ship_id state
    :param raw_state: dictionary of 2d ndarray
    :param me_id: int
    :return: 3d ndarray
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
    processed_state.append((ship_map == me_id) * 1.) # 0
    processed_state.append((ship_map != me_id) * 1.) # 1
    # one hot shipyard state
    processed_state.append((shipyard_map == me_id) * 1.) # 2
    processed_state.append((shipyard_map != me_id) * 1.) # 3
    # one hot dropoff state
    processed_state.append((dropoff_map == me_id) * 1.) # 4
    processed_state.append((dropoff_map != me_id) * 1.) # 5
    # ship halite state
    processed_state.append((ship_map == me_id) * ship_halite_map / constants.MAX_HALITE) # 6
    processed_state.append((ship_map != me_id) * ship_halite_map / constants.MAX_HALITE) # 7
    # halite state
    processed_state.append(halite_map / constants.MAX_HALITE) # 8
    # moving cost state
    processed_state.append(np.trunc(np.maximum(halite_map / constants.MOVE_COST_RATIO, 1) - ((halite_map == 0) * 1.)) / constants.MAX_HALITE) # 9
    # remaining round
    processed_state.append(np.ones(ship_map.shape) * (constants.MAX_TURNS - turn_number))  # 10

    # todo add inspired halite / ship state / move cost

    # ship id state (this is used to center the matrix for different ship)
    processed_state.append( (-np.ones(ship_id_map.shape) * (ship_map != me_id)) + ship_id_map * (ship_map == me_id)) # +1

    return np.array(processed_state)


def center_state_for_ship(state, ship_id_map, ship_id, shape=None):
    """
    center the state from ship perspective
    :param state: 3d ndarray
    :param ship_id_map: 2d ndarray
    :param ship_id: int
    :param shape: None or int
    :return: 3d ndarray
    """

    _ship_location = np.where(ship_id_map == ship_id)
    _center_location = ((np.array(state[0].shape) - 1) / 2).astype(int)

    if len(_ship_location[0]) != 1:
        print('function center_state_for_ship: more then one / zero match found for ship_id {}, ship location {}'.format(ship_id, _ship_location))
        return

    centered_state = state

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


def get_halite_command(turn_limit=300, replay_directory='', parameters = {}):
    cmd = ["./halite", "--width 32", "--height 32", "--no-timeout", "--no-compression"]
    cmd.append("--turn-limit %d" % turn_limit)
    if replay_directory:
        cmd.append("--replay-directory %s" % (replay_directory))
        cmd.append("-vvv")
    cmd.append("python3 8_.py --MAX_SHIP_ON_MAP 0 --COLLISION_2P 0 --MAKE_DROPOFF_GAIN_COST_RATIO 999")

    script = "python3 rl_1.py"
    for p in parameters.keys():
        script += ' --%s %s' % (p, parameters[p])

    cmd.append(script)
    return cmd


# parameter
_parameters = {
    'epsilon_train': 0.01,
    'epsilon_decay_period': 250000,
    'batch_size': 32,
    'discount': 0.99,
    'min_replay_history': 20000,
    'target_update_period': 8000,
    'training_steps': 0,
    'folder': 'simple_dqn'
}

max_training_steps = 5000000
games_num = 0
actions_list = np.array([commands.NORTH, commands.EAST, commands.SOUTH, commands.WEST, commands.STAY_STILL])

if __name__ == "__main__":
    print('simple_dqn begin')

    # Create folder
    if not os.path.exists(_parameters['folder']):
        os.mkdir(_parameters['folder'])

    # Set training_steps
    if os.path.exists(os.path.join(_parameters['folder'], 'state_replay.npy')):
        state_replay = np.load(os.path.join(_parameters['folder'], 'state_replay.npy'))
        _parameters['training_steps'] = len(state_replay)
        del state_replay

    # Init q_network and target_network
    if not os.path.exists(os.path.join(_parameters['folder'], 'q_network')):
        q_network = DQN(output=len(actions_list))
        target_network = DQN(output=len(actions_list))
        target_network.model.set_weights(q_network.model.get_weights())
        q_network.model.save(os.path.join(_parameters['folder'], 'q_network'))
        target_network.model.save(os.path.join(_parameters['folder'], 'target_network'))

    # Start experiment
    while _parameters['training_steps'] < max_training_steps:
        turn_limit = np.random.randint(2, 12) * 50 # 100, 150, ..., 550
        replay_directory = os.path.join(_parameters['folder'], datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(replay_directory):
            os.mkdir(replay_directory)
        cmd = get_halite_command(replay_directory = replay_directory, turn_limit = turn_limit + 2, parameters = _parameters)

        games_num += 1
        print('\nsimple_dqn starting %d game, trained %d steps\n' % (games_num, _parameters['training_steps']))
        subprocess.call(cmd)
        subprocess.call(["mv", "bot-0.log", "./{}/{}".format(replay_directory, "bot-0.log")])
        subprocess.call(["mv", "bot-1.log", "./{}/{}".format(replay_directory, "bot-1.log")])

        _parameters['training_steps'] += turn_limit

    print('simple_dqn done')







