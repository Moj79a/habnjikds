import gymnasium as gym
import panda_gym
import pickle
from numpngw import write_apng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import DDPG, TD3 , SAC
from stable_baselines3.td3.policies import MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

#env = gym.make("PandaReach1-v3", render_mode="human" , reward_type="dense" , control_type="joints")

log_dir_sac = "/home/moj79/panda_gym6/log/sac"
os.makedirs(log_dir_sac, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
monitor_path = os.path.join(log_dir_sac, f"monitor_{run_id}.csv")
log_filepath = "observer_log_cumulative.csv" # تعریف مسیر فایل لاگ رویتگر

if os.path.exists(log_filepath):
    os.remove(log_filepath)
    print(f"Removed old log file: {log_filepath}")
        

def make_env():
    env = gym.make("PandaReach2-v3", render_mode="human", reward_type="dense", control_type="joints")
    env = Monitor(env, monitor_path, info_keywords=("is_success",))
    return env

vec_env = DummyVecEnv([make_env])
vec_env.reset()
        
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` episodes)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int) Number of episodes between each check.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.episode_count = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.episode_count += 1

        if self.episode_count % self.check_freq == 0:

            x, y = ts2xy(load_results(self.log_dir), "episodes")
            if len(x) > 0:
                mean_reward = np.mean(y[-self.check_freq:])
                if self.verbose > 0:
                    print(f"\nNum episodes: {self.episode_count}")
                    print(f"Num timesteps: {self.n_calls}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    
                    with open(os.path.join(self.log_dir, "replay_buffer.pkl"), "wb") as f:
                        save_partial_replay_buffer(self.model.replay_buffer, os.path.join(self.log_dir, "replay_buffer.pkl"), max_size=100000)
        return True

def save_partial_replay_buffer(buffer, filepath, max_size=100000):
    size = buffer.size() if callable(buffer.size) else buffer.size
    start_idx = max(0, size - max_size)

    # بخش observations
    if isinstance(buffer.observations, dict):
        observations = {}
        for key, value in buffer.observations.items():
            observations[key] = value[start_idx:size].copy()
    else:
        observations = buffer.observations[start_idx:size].copy()

    # بخش next_observations
    if isinstance(buffer.next_observations, dict):
        next_observations = {}
        for key, value in buffer.next_observations.items():
            next_observations[key] = value[start_idx:size].copy()
    else:
        next_observations = buffer.next_observations[start_idx:size].copy()

    partial_data = {
        'observations': observations,
        'actions': buffer.actions[start_idx:size].copy(),
        'next_observations': next_observations,
        'rewards': buffer.rewards[start_idx:size].copy(),
        'dones': buffer.dones[start_idx:size].copy(),
        'size': size - start_idx,
        'pos': size - start_idx,
        'full': (size - start_idx) == buffer.buffer_size,
        'buffer_size': buffer.buffer_size
    }
    with open(filepath, 'wb') as f:
        pickle.dump(partial_data, f)


# ۳. کال‌بک برای ذخیره بهترین مدل
check_freq = 10
callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir_sac)

# ۴. تنظیمات کلی آموزش
total_timesteps = 20000000

# ۶. تعریف مدل SAC
model = SAC(
    "MultiInputPolicy",           # چون env شما observation چندجزیی دارد
    vec_env,
    #DummyVecEnv([lambda: log_env]),
    learning_rate=1e-4,           # نرخ یادگیری
    buffer_size=int(1e7),
    batch_size=256,               # اندازه دسته بهینه‌سازی
    #train_freq=(1, "episode"),    # به‌روزرسانی پس از هر اپیزود
    gamma=0.95,                   # ضریب تنزیل
    verbose=2,
    #device="cpu",
)

# ۷. شروع یادگیری
model.learn(
    total_timesteps=total_timesteps,
    log_interval=50,
    callback=callback,
)
model.save(os.path.join(log_dir_sac, "best_model"))

# --------------------------------------------------
# تحلیل و رسم نمودار
print("\nTraining/Simulation finished. Processing logged data...")

try:
    df = pd.read_csv(log_filepath)
    print(f"Successfully loaded {len(df)} rows of data from '{log_filepath}'.")
except FileNotFoundError:
    print(f"ERROR: Log file not found at '{log_filepath}'. Cannot generate plots.")
    df = pd.DataFrame()
except Exception as e:
    print(f"Error reading log file: {e}")
    df = pd.DataFrame()

if not df.empty:
    num_movable_joints = 7

    # --- اصلاح زمان به صورت پیوسته ---
    df = df.sort_values(by=['episode', 'time']).reset_index(drop=True)
    df['continuous_time'] = 0.0
    time_offset = 0.0

    for ep in df['episode'].unique():
        ep_mask = df['episode'] == ep
        ep_times = df.loc[ep_mask, 'time']
        df.loc[ep_mask, 'continuous_time'] = ep_times + time_offset
        time_offset = df.loc[ep_mask, 'continuous_time'].iloc[-1]

    dt = df['continuous_time'].diff().mean()

    plt.style.use('seaborn-v0_8-whitegrid')

    # نمودار مقایسه سرعت‌ها
    fig_vel, axes_vel = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
    axes_vel = axes_vel.flatten()
    for j in range(num_movable_joints):
        ax = axes_vel[j]
        ax.plot(df['continuous_time'], df[f'vel_true_j{j}'], label='True Velocity', linewidth=2)
        ax.plot(df['continuous_time'], df[f'vel_est_j{j}'], label='Estimated Velocity', linestyle='--', linewidth=2)
        ax.set_title(f'Joint {j} Velocity')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
    if len(axes_vel) > num_movable_joints: fig_vel.delaxes(axes_vel[-1])
    fig_vel.suptitle('Observer Performance: Velocity Comparison (Continuous Time)', fontsize=16)
    fig_vel.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('observer_velocity_comparison_continuous.png')
    print("Velocity comparison plot saved.")

    # نمودار مقایسه موقعیت‌ها
    fig_pos, axes_pos = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
    axes_pos = axes_pos.flatten()
    for j in range(num_movable_joints):
        ax = axes_pos[j]
        ax.plot(df['continuous_time'], df[f'pos_true_j{j}'], label='True Position', linewidth=2)
        ax.plot(df['continuous_time'], df[f'pos_est_j{j}'], label='Estimated Position', linestyle='--', linewidth=2)
        ax.set_title(f'Joint {j} Position')
        ax.set_ylabel('Position (rad)')
        ax.legend()
    if len(axes_pos) > num_movable_joints: fig_pos.delaxes(axes_pos[-1])
    fig_pos.suptitle('Observer Performance: Position Comparison (Continuous Time)', fontsize=16)
    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('observer_position_comparison_continuous.png')
    print("Position comparison plot saved.")

    # نمودار معیارهای خطا
    fig_err, axes_err = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    for j in range(num_movable_joints):
        vel_error = df[f'vel_true_j{j}'] - df[f'vel_est_j{j}']
        ise_cumulative = (vel_error**2).cumsum() * dt
        mae_cumulative = vel_error.abs().expanding().mean()
        rmse_cumulative = ((vel_error**2).expanding().mean())**0.5
        axes_err[0].plot(df['continuous_time'], ise_cumulative, label=f'Joint {j}')
        axes_err[1].plot(df['continuous_time'], mae_cumulative, label=f'Joint {j}')
        axes_err[2].plot(df['continuous_time'], rmse_cumulative, label=f'Joint {j}')
    axes_err[0].set_title('Integral Squared Error (ISE) - Velocity'); axes_err[0].legend()
    axes_err[1].set_title('Mean Absolute Error (MAE) - Velocity'); axes_err[1].legend()
    axes_err[2].set_title('Root Mean Squared Error (RMSE) - Velocity'); axes_err[2].legend()
    axes_err[2].set_xlabel("Continuous Time (s)")
    plt.savefig('observer_error_metrics_continuous.png')
    print("Error metrics plot saved.")

    plt.show()


# --------------------------------------------------
# بستن محیط
vec_env.close()
