import os
import numpy as np
import gymnasium as gym


def replay_episodes(folder_path, env_name):
  """
  Replay all saved episodes in the specified folder in the given Gym environment.

  :param folder_path: Path to the folder containing the saved episodes (.npz files).
  :param env_name: Name of the Gym environment to use for replay.
  """
  # Initialize the environment
  env = gym.make(env_name, render_mode='human')

  # List all .npz files in the folder
  episode_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

  if not episode_files:
    print("No episodes found in the folder.")
    return

  for episode_file in sorted(episode_files):
    episode_path = os.path.join(folder_path, episode_file)
    data = np.load(episode_path)

    states = data['states']
    actions = data['actions']

    print(f"Replaying episode: {episode_file}")
    env.reset()

    for step, (state, action) in enumerate(zip(states, actions)):
      print(f"Step {step}: State: {state}, Action: {action}")
      _, _, terminated, truncated, _ = env.step(action)

      if terminated or truncated:
        break

    print(f"Finished replaying episode: {episode_file}")

  env.close()


replay_episodes('./episodes', 'LunarLander-v3')
