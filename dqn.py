import tensorflow as tf
import gym
import numpy as np
import random

from collections import deque, namedtuple
from env import wrap_dqn, ScaledFloatFrame
from util import *

class DQN(object):
    def __init__(self, num_actions, summary_writer):
        self._build_graph(num_actions)

    def _build_graph(self, num_actions):
        # the model
        with tf.variable_scope('main'):
            self.x = x = tf.placeholder(tf.float32, [None, 84, 84, 4], name="input")
            self.batch_size = tf.shape(x)[0]
            # convolution layer
            x = tf.nn.relu( conv2d(x, 32, "l1", [8,8], [4,4]) )
            x = tf.nn.relu( conv2d(x, 64, "l2", [4,4], [2,2]) )
            x = tf.nn.relu( conv2d(x, 64, "l3", [3,3], [1,1]) )
            # fully connected layer
            x = tf.nn.relu(linear(flatten(x), 512, "hidden", normalized_columns_initializer(1.0)))
            self.q_values = linear(x, num_actions, "q_out", normalized_columns_initializer(1.0))
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        with tf.variable_scope('target'):
            self.x_target = x = tf.placeholder(tf.float32, [None, 84, 84, 4], name="input_target")
            # convolution layer
            x = tf.nn.relu( conv2d(x, 32, "l1", [8,8], [4,4]) )
            x = tf.nn.relu( conv2d(x, 64, "l2", [4,4], [2,2]) )
            x = tf.nn.relu( conv2d(x, 64, "l3", [3,3], [1,1]) )
            # fully connected layer
            x = tf.nn.relu(linear(flatten(x), 512, "hidden", normalized_columns_initializer(1.0)))
            self.q_values_target = linear(x, num_actions, "q_out", normalized_columns_initializer(1.0))
            self.var_list_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self.sync = tf.group(
                    *(
                        [v1.assign(v2) for v1, v2 in zip(self.var_list_target, self.var_list)]
                    ))

        # act
        self.eps = tf.placeholder(tf.float32, [1])
        deterministic_actions = tf.argmax(self.q_values, axis=1)
        random_actions = tf.random_uniform(tf.stack([self.batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([self.batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps[0]
        self.stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        # train
        self.rewards_t = tf.placeholder(tf.float32, [None], name="reward")
        self.actions_t = tf.placeholder(tf.int32, [None], name="action")
        self.done_mask = tf.placeholder(tf.float32, [None], name="done")

        # q scores for actions, we know were selected
        q_t_selected = tf.reduce_sum(self.q_values * tf.one_hot(self.actions_t, num_actions), 1)

        q_tp1_best = tf.reduce_max(self.q_values_target, 1)
        q_tp1_best_masked = (1.0 - self.done_mask) * q_tp1_best

        # compute RHS of bellman equation
        gamma = 0.99
        q_t_selected_target = self.rewards_t + gamma * q_tp1_best_masked

        # compute error
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

        # huber loss
        delta = 1.0
        errors = tf.where(tf.abs(td_error) < delta,
                          tf.square(td_error) * 0.5,
                          delta * (tf.abs(td_error) - 0.5 * delta))

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.optimize_expr = optimizer.minimize(errors)

    def act(self, obs, epsilon):
        # sample an action
        sess = tf.get_default_session()
        return sess.run([self.stochastic_actions],
                        feed_dict={self.x : obs, self.eps : epsilon})

    def train(self, obses, actions, rewards, obses_tp1, dones):
        # train
        sess = tf.get_default_session()
        return sess.run([self.optimize_expr],
                        feed_dict={self.x : obses,
                                   self.actions_t : actions,
                                   self.rewards_t : rewards,
                                   self.x_target : obses_tp1,
                                   self.done_mask : dones
                                  })

    def update_target(self):
        sess = tf.get_default_session()
        return sess.run(self.sync)


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def learn(env,
          model,
          summary_writer,
          batch_size=32,
          max_timesteps = 100000000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=4,
          learning_starts=10000,
          target_network_update_freq=1000,
          buffer_size=50000
          ):
    """
    Training
    """

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Create the replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Initial setup
    episode_reward = 0.0
    episode_shaped_reward = 0.0
    episode_length = 0
    obs = env.reset()

    for t in range(max_timesteps):

        # choose action
        action = model.act([obs], [exploration.value(t)])[0][0]
        # act on env
        new_obs, reward, done, _ = env.step(action)
        # clip reward
        shaped_reward = min(1, max(-1, reward))
        # Store transition in the replay buffer
        replay_buffer.add(obs, action, shaped_reward, new_obs, float(done))
        obs = new_obs

        episode_reward += reward
        episode_shaped_reward += shaped_reward
        episode_length += 1
        if done:
            # episode ended
            obs = env.reset()
            # summary
            summary = tf.Summary()
            summary.value.add(tag='global/episode_reward', simple_value=episode_reward)
            summary.value.add(tag='global/episode_shaped_reward', simple_value=episode_shaped_reward)
            summary.value.add(tag='global/episode_length', simple_value=episode_length)
            summary_writer.add_summary(summary, global_step=t)
            # reset episode_reward
            episode_reward = 0.0
            episode_shaped_reward = 0.0
            episode_length = 0

        if t > learning_starts and t % train_freq == 0:
            # Train network periodically
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            model.train(obses_t, actions, rewards, obses_tp1, dones)


        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

    return None



if __name__ == "__main__":
    with tf.Session() as sess:
        # Build environment
        env = gym.make("PongNoFrameskip-v3")

        # Pre-processing
        env = ScaledFloatFrame(wrap_dqn(env))

        # Summary writer
        name = "Pong"
        summary_writer = tf.summary.FileWriter('experiments/' + name + '/')

        # Build model's graph
        model = DQN(env.action_space.n, summary_writer)

        # Initialise model
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize() # Finalise graph

        # Trian the model
        learn(env, model, summary_writer)

        # close
        summary_writer.close()
        env.close()
