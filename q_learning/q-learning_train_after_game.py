import gym
import numpy as np
import random
import pandas as pd

# Init Taxi-V2 Env
env = gym.make("Taxi-v2").env

# Init arbitary values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
max_training_steps = 1000000
alpha = 0.3
discount = 0.6
epsilon_train = 0.01
epsilon_decay_period = max_training_steps//2

all_epochs = []
all_penalties = []

# episode_rewards = pd.DataFrame([], columns=['avg_rewards','total_rewards','max_rewards','epsilon','avg_loss','total_loss','max_loss','avg_q','total_q','max_q'])
episode_rewards = {
    'avg_rewards': [],
    'total_rewards': [],
    'max_rewards': [],
    'epsilon': [],
    'avg_loss': [],
    'total_loss': [],
    'max_loss': [],
    'avg_q': [],
    'total_q': [],
    'max_q': [],
}

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


for i in range(1, max_training_steps+1):
    state = env.reset()

    # Init Vars
    epochs, penalties, reward, = 0, 0, 0
    done = False
    _reward_list = []
    _loss_list = []
    _q_list = []
    _state_list = []
    _next_state_list = []
    _action_list = []


    while not done:
        epsilon = linearly_decaying_epsilon(i-1, 0, epsilon_decay_period, epsilon_train)
        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()
        else:
            # Check the learned values
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        # old_value = q_table[state, action]
        # next_max = np.max(q_table[next_state])

        _reward_list.append(reward)
        _action_list.append(action)
        _state_list.append(state)
        _next_state_list.append(next_state)

        # Update the new value
        # new_value = (1 - alpha) * old_value + alpha * (reward + discount * next_max)
        # q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    for j in range(len(_state_list)):
        state, action, reward, next_state = _state_list[j], _action_list[j], _reward_list[j], _next_state_list[j]

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + discount * next_max)
        q_table[state, action] = new_value

        _loss_list.append(abs(reward + discount * next_max - old_value))
        _q_list.append(old_value)

    episode_rewards['avg_rewards'].append(np.array(_reward_list).mean())
    episode_rewards['total_rewards'].append(np.array(_reward_list).sum())
    episode_rewards['max_rewards'].append(np.array(_reward_list).max())
    episode_rewards['epsilon'].append(epsilon)
    episode_rewards['avg_loss'].append(np.array(_loss_list).mean())
    episode_rewards['total_loss'].append(np.array(_loss_list).sum())
    episode_rewards['max_loss'].append(np.array(_loss_list).max())
    episode_rewards['avg_q'].append(np.array(_q_list).mean())
    episode_rewards['total_q'].append(np.array(_q_list).sum())
    episode_rewards['max_q'].append(np.array(_q_list).max())

    if i % 100 == 0:
        print('episode=%d\trewards: sum=%.4f\tmean=%.4f' % (i, np.array(_reward_list).sum(), np.array(_reward_list).mean()), flush=True)
np.save('q_table_train_after_game.npy', q_table)
episode_rewards = pd.DataFrame.from_dict(episode_rewards)
episode_rewards.to_parquet('episode_log_train_after_game.snappy.parquet')
print("Training finished.\n")