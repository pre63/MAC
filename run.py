import mac
import numpy
import gymnasium as gym
import numpy as np
import sys
import random
import tensorflow as tf
import utils

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# get and set hyper-parameters
meta_params, alg_params = {}, {}
try:
  meta_params['env_name'] = sys.argv[1]
  meta_params['seed_number'] = int(sys.argv[2])
except:
  print("default environment is Lunar Lander ...")
  meta_params['env_name'] = 'LunarLander-v3'
  meta_params['seed_number'] = 0

meta_params['env'] = gym.make(meta_params['env_name'])

if meta_params['env_name'] == 'CartPole-v0':
  meta_params['max_learning_episodes'] = 200
  meta_params['gamma'] = 0.9999
  meta_params['plot'] = False
  alg_params = {}
  alg_params['|A|'] = meta_params['env'].action_space.n
  alg_params['state_|dimension|'] = len(meta_params['env'].reset())
  alg_params['state_|dimension|'] = meta_params['env'].observation_space.shape[0]

  alg_params['critic_num_h'] = 1
  alg_params['critic_|h|'] = 64
  alg_params['critic_lr'] = 0.01
  alg_params['actor_num_h'] = 1
  alg_params['actor_|h|'] = 64
  alg_params['actor_lr'] = 0.005
  alg_params['critic_batch_size'] = 32
  alg_params['critic_num_epochs'] = 10
  alg_params['critic_target_net_freq'] = 1
  alg_params['max_buffer_size'] = 2000
  alg_params['critic_train_type'] = 'model_free_critic_TD'  # or model_free_critic_monte_carlo

if meta_params['env_name'] == 'LunarLander-v3':
  meta_params['max_learning_episodes'] = 3000
  meta_params['gamma'] = 0.9999
  meta_params['plot'] = False
  alg_params = {}
  alg_params['|A|'] = meta_params['env'].action_space.n
  alg_params['state_|dimension|'] = len(meta_params['env'].reset())
  alg_params['state_|dimension|'] = meta_params['env'].observation_space.shape[0]

  alg_params['critic_num_h'] = 1
  alg_params['critic_|h|'] = 64
  alg_params['critic_lr'] = 0.005
  alg_params['actor_num_h'] = 1
  alg_params['actor_|h|'] = 64
  alg_params['actor_lr'] = 0.0005
  alg_params['critic_batch_size'] = 32
  alg_params['critic_num_epochs'] = 10
  alg_params['critic_target_net_freq'] = 1
  alg_params['max_buffer_size'] = 5000
  alg_params['critic_train_type'] = 'model_free_critic_TD'  # or model_free_critic_monte_carlo


# Ensure results are reproducible
np.random.seed(meta_params['seed_number'])
random.seed(meta_params['seed_number'])

# TensorFlow 2.x uses eager execution, so you don't need a session unless for compatibility
tf.random.set_seed(meta_params['seed_number'])

# Create the environment
meta_params['env_name'] = 'LunarLander-v3'  # Use the updated environment version
meta_params['env'].reset(seed=meta_params['seed_number'])

# Configure Keras backend (if necessary)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
  for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Create and train a MAC agent
agent = mac.mac(alg_params)
agent.train(meta_params)
