import gymnasium as gym
import panda_gym
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import time
import numpy as np

# --- مسیر لاگ اولیه و مسیر نسخه‌بندی ادامه ---
log_dir_initial = "/home/moj79/panda_gym2/log/sac_initial"
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir_continue = f"/home/moj79/panda_gym2/log/sac_continue_{timestamp}"
os.makedirs(log_dir_continue, exist_ok=True)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.episode_count = 0

    def _init_callback(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.episode_count += 1
        if self.episode_count % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "episodes")
            if len(x) > 0:
                mean_reward = y[-self.check_freq:].mean()
                if self.verbose > 0:
                    print(f"[Callback] Episodes: {self.episode_count} | Mean reward: {mean_reward:.2f} | Best reward: {self.best_mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    with open(os.path.join(self.log_dir, "replay_buffer.pkl"), "wb") as f:
                        pickle.dump(self.model.replay_buffer, f)
        return True

callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir_continue)

def make_env():
    env = gym.make("PandaReach1-v3", render_mode="human", reward_type="dense", control_type="joints")
    env = Monitor(env, log_dir_continue, info_keywords=("is_success",))
    return env

vec_env = DummyVecEnv([make_env])

model = SAC.load(os.path.join(log_dir_initial, "best_model"), env=vec_env)

replay_buffer_path = os.path.join(log_dir_initial, "replay_buffer.pkl")
if os.path.exists(replay_buffer_path):
    with open(replay_buffer_path, "rb") as f:
        model.replay_buffer = pickle.load(f)
    print("Replay buffer loaded successfully.")
else:
    print("Replay buffer file not found. Starting with empty buffer.")

model.learn(total_timesteps=6_000_000, reset_num_timesteps=False, log_interval=50, callback=callback)

model.save(os.path.join(log_dir_continue, "best_model"))
with open(os.path.join(log_dir_continue, "replay_buffer.pkl"), "wb") as f:
    pickle.dump(model.replay_buffer, f)

