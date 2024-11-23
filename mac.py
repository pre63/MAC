import numpy as np
import critic_network
import actor_network
import gymnasium as gym
import sys
import utils
import os
from collections import deque


class mac:
  """
  A class representing the Mean Actor-Critic algorithm.
  It contains an actor, a critic, and a train function.
  """

  def __init__(self, params):
    """
    Initializes the MAC agent by creating an actor and a critic.
    """
    self.params = params
    self.memory = deque(maxlen=self.params['max_buffer_size'])
    self.actor = actor_network.actor(self.params)
    self.critic = critic_network.critic(self.params)

  def add_to_memory(self, states, actions, rewards):
    T = len(states)
    for index, (s, a, r) in enumerate(zip(states, actions, rewards)):
      if index < T - 1:
        self.memory.append((s, a, r, states[index + 1], T - index - 1))

  def train(self, meta_params):
    """
    Trains a MAC agent for max_learning_episodes episodes.
    Proceeds with interaction for one episode, followed by critic
    and actor updates.
    """
    print("Training has begun...")
    li_episode_length = []
    li_returns = []

    for episode in range(1, meta_params['max_learning_episodes']):
      states, actions, returns, rewards = self.interact_one_episode(meta_params, episode)
      self.add_to_memory(states, actions, rewards)

      # Log performance
      li_episode_length.append(len(states))
      if episode % 10 == 0:
        print(episode, "return in last 10 episodes", np.mean(li_returns[-10:]))
      li_returns.append(returns[0])
      sys.stdout.flush()

      # Train the Q network
      if self.params['critic_train_type'] == 'model_free_critic_monte_carlo':
        self.critic.train_model_free_monte_carlo(states, actions, returns)
      elif self.params['critic_train_type'] == 'model_free_critic_TD':
        self.critic.train_model_free_TD(self.memory, self.actor, meta_params, self.params, episode)
      self.actor.train(states, self.critic)

    print("Training is finished successfully!")
    return

  def interact_one_episode(self, meta_params, episode):
    """
    Execute one episode of interaction between the MAC agent and the environment.
    Returns states, actions, returns, and rewards for training.
    """
    print("Episode: ", episode)
    # Initialize variables
    s0, info = meta_params['env'].reset(seed=meta_params['seed_number'])
    rewards, states, actions = [], [], []
    s = s0

    while True:
      # Select action using the actor
      a = self.actor.select_action(s)

      # Perform action in the environment
      s_p, r, terminated, truncated, _ = meta_params['env'].step(a)

      # Store states, actions, and rewards
      states.append(s)
      actions.append(a)
      rewards.append(r)

      # Check for end of episode
      if terminated or truncated:
        # Store terminal state and a dummy action
        states.append(s_p)
        a = self.actor.select_action(s_p)
        actions.append(a)
        rewards.append(0)  # Append a zero reward for terminal state
        break

      # Update state
      s = s_p

    # Compute returns from rewards
    returns = utils.rewardToReturn(rewards, meta_params['gamma'])

    return states, actions, returns, rewards
