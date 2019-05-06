import gym
from simple_dqn import *
import pickle


class DQN(object):

    def __init__(self, input=(11, 32, 32), output=5):
        self.input = input
        self.output = output

        self.model = Sequential()
        self.model.add(Conv2D(3, (3, 3), strides=(1, 1), activation="relu", input_shape=self.input, data_format='channels_last'))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
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


class ReplayBuffer(object):
    def __init__(self, max_size):
        """Simple replay buffer for storing sampled DQN (s, a, s', r) transitions as tuples.

        :param size: Maximum size of the replay buffer.
        """
        self._buffer = []
        self._max_size = max_size
        self._idx = 0

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, act, rew, obs_tp1, done):
        """
        Add a new sample to the replay buffer.
        :param obs_t: observation at time t
        :param act:  action
        :param rew: reward
        :param obs_tp1: observation at time t+1
        :param done: termination signal (whether episode has finished or not)
        """
        data = (obs_t, act, rew, obs_tp1, done)
        if self._idx >= len(self._buffer):
            self._buffer.append(data)
        else:
            self._buffer[self._idx] = data
        self._idx = (self._idx + 1) % self._max_size

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._buffer[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of transition tuples.

        :param batch_size: Number of sampled transition tuples.
        :return: Tuple of transitions.
        """
        idxes = [random.randint(0, len(self._buffer) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def dump(self, file_path=None):
        """Dump the replay buffer into a file.
        """
        file = open(file_path, 'wb')
        pickle.dump(self._buffer, file, -1)
        file.close()

    def load(self, file_path=None):
        """Load the replay buffer from a file
        """
        file = open(file_path, 'rb')
        self._buffer = pickle.load(file)
        file.close()


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


def preprocess(raw_state, game):
    if game == 'CubeCrash-v0':
        processed_state = raw_state[:,:,1:2] / 255
    else:
        processed_state = raw_state
    return processed_state


if __name__ == "__main__":
    import argparse, os, json, pandas as pd

    parser = argparse.ArgumentParser(__name__)
    # General
    parser.add_argument("--folder", default='atari_test', type=str, help="name of folder to store everything")
    parser.add_argument("--game", default='CubeCrash-v0', type=str, help="game to play")
    parser.add_argument("--suffix", default='', type=str, help="suffix for all files stored")
    parser.add_argument("--max_training_steps", default=20000, type=int, help="max training steps")
    parser.add_argument("--double", default=0, type=int, help="use double q network or not")
    parser.add_argument("--eval", default=0, type=int, help="evaluation mode")
    args = parser.parse_args()

    _q_network_name = 'q_network' + args.suffix
    _target_network_name = 'target_network' + args.suffix
    _buffer_name = 'buffer' + args.suffix
    _config_name = 'config' + args.suffix
    _training_steps_name = 'training_steps' + args.suffix
    _episode_rewards_name = 'episode_rewards' + args.suffix + '.snappy.parquet'
    _folder = os.path.join(args.folder, args.game)
    _game = args.game
    if not os.path.exists(_folder): os.mkdir(_folder)

    eval = args.eval
    max_training_steps = args.max_training_steps
    replay_capacity = 5000
    min_replay_history = 2000
    batch_size = 32
    epsilon_decay_period = 25000
    epsilon_train = 0.01
    epsilon_eval = 0.0
    init_training_steps = 0
    discount = 0.99
    target_update_period = 800
    double = args.double

    env = gym.make(_game)
    state_size = preprocess(env.reset(), _game).shape
    action_size = env.action_space.n
    q_network = DQN(state_size, action_size)
    target_network = DQN(state_size, action_size)
    target_network.set_weights(q_network.get_weights())
    episode_rewards = pd.DataFrame([], columns=['avg_rewards','total_rewards','max_rewards','epsilon'])
    done = False

    buffer = ReplayBuffer(replay_capacity)

    if os.path.exists(os.path.join(_folder, _q_network_name)):
        q_network.load_model(os.path.join(_folder,_q_network_name))
    if os.path.exists(os.path.join(_folder,_target_network_name)):
        target_network.load_model(os.path.join(_folder,_target_network_name))
    if os.path.exists(os.path.join(_folder,_buffer_name)):
        buffer.load(os.path.join(_folder,_buffer_name))
    if os.path.exists(os.path.join(_folder,_episode_rewards_name)):
        episode_rewards = pd.read_parquet(os.path.join(_folder,_episode_rewards_name))
    if os.path.exists(os.path.join(_folder, _training_steps_name)):
        f = open(os.path.join(_folder, _training_steps_name), 'r')
        init_training_steps = int(f.readline())
        f.close()

    training_steps = init_training_steps
    if eval:
        from matplotlib import pyplot as plt
        state = env.reset()
        _reward_list = []
        for i in range(100):
            q_val = q_network.predict(preprocess(np.array([state])))
            action = np.argmax(q_val)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            print(reward, action, q_val, flush=True)
            plt.imshow(next_state)
            plt.show()
            time.sleep(0.05)
            _reward_list.append(reward)
            if done: break
        print('mean reward = {}', np.array(_reward_list).mean(), flush=True)
    else:
        while training_steps < max_training_steps:
            state = env.reset()
            state = preprocess(state, _game)
            _reward_list = []
            print('----reset----', flush=True)
            while True:
                # Select actions
                epsilon = linearly_decaying_epsilon(training_steps, min_replay_history, epsilon_decay_period, epsilon_train) if not eval else epsilon_eval
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_size)
                else:
                    action = np.argmax(q_network.predict(np.array([state])))

                # Run experiment
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess(next_state, _game)
                _reward_list.append(reward)
                # print(reward, flush=True)

                # Save experience
                buffer.add(state, action, reward, next_state, done)
                state = next_state

                if len(buffer) >= min_replay_history:
                    # Fit network
                    _s, _a, _r, _n, _d = buffer.sample(batch_size)
                    __a = np.zeros((_a.shape[0], action_size))
                    __a[np.arange(_a.shape[0]), _a] = 1
                    _a = __a # 2d array
                    q_network.fit(_s, _a, _r, _n, _d, double=double, discount=discount, target_network=target_network)

                    # Sync target network
                    if training_steps % target_update_period == 0:
                        target_network.set_weights(q_network.get_weights())

                training_steps += 1
                if done:
                    print('mean reward = {}', np.array(_reward_list).mean(), flush=True)
                    episode_rewards = episode_rewards.append(
                        {
                            'avg_rewards' : np.array(_reward_list).mean(),
                            'total_rewards' : np.array(_reward_list).sum(),
                            'max_rewards': np.array(_reward_list).max(),
                            'epsilon': epsilon,
                        }, ignore_index=True)
                    break

        q_network.save(os.path.join(_folder, _q_network_name))
        target_network.save(os.path.join(_folder, _target_network_name))
        buffer.dump(os.path.join(_folder, _buffer_name))
        episode_rewards.to_parquet(os.path.join(_folder, _episode_rewards_name))
        with open(os.path.join(_folder, _training_steps_name), 'w') as f:
            f.write('%d' % training_steps)
        with open(os.path.join(_folder, _config_name), 'a') as outfile:
            json.dump({
                'eval': eval,
                'max_training_steps': max_training_steps,
                'replay_capacity': replay_capacity,
                'batch_size': batch_size,
                'epsilon_decay_period': epsilon_decay_period,
                'epsilon_train': epsilon_train,
                'epsilon_eval': epsilon_eval,
                'init_training_steps': init_training_steps,
                'discount': discount,
                'target_update_period': target_update_period,
            }, outfile, sort_keys=True, indent=4)

# eval
# state = env.reset()
# # q_network.load_model('q_network')
# for i in range(100):
#     q_val = q_network.predict(np.array([state]))
#     action = np.argmax(q_val)
#     next_state, reward, done, _ = env.step(action)
#     state = next_state
#     print(reward, action, q_val)
#     plt.imshow(next_state)
#     plt.show()
#     time.sleep(0.05)
#     if done: break

