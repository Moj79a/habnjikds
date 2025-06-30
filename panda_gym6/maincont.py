import os
import gymnasium as gym
import panda_gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from datetime import datetime

# --------------------------------------------------
# ۱) مسیرها و تنظیمات
log_dir = "/home/moj79/panda_gym6/log/sac"
best_model = os.path.join(log_dir, "best_model.zip")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(log_dir, exist_ok=True)
monitor_path = os.path.join(log_dir, f"monitor_{run_id}.csv")
log_filepath = "observer_log_cumulative.csv"
total_timesteps_target = 2000

# === پاک کردن فایل لاگ قدیمی قبل از اجرای برنامه ===
if os.path.exists(log_filepath):
    os.remove(log_filepath)
    print(f"Removed old log file: {log_filepath}")

# --------------------------------------------------
# توابع ذخیره و بازیابی بافر (کامل و بدون تغییر)
def save_partial_replay_buffer(buffer, filepath, max_size=100000):
    size = buffer.size
    start_idx = max(0, size - max_size)
    if isinstance(buffer.observations, dict):
        observations = {k: v[start_idx:size].copy() for k, v in buffer.observations.items()}
    else:
        observations = buffer.observations[start_idx:size].copy()
    if isinstance(buffer.next_observations, dict):
        next_observations = {k: v[start_idx:size].copy() for k, v in buffer.next_observations.items()}
    else:
        next_observations = buffer.next_observations[start_idx:size].copy()
    partial_data = {
        'observations': observations, 'actions': buffer.actions[start_idx:size].copy(),
        'next_observations': next_observations, 'rewards': buffer.rewards[start_idx:size].copy(),
        'dones': buffer.dones[start_idx:size].copy(), 'size': size - start_idx,
        'pos': size - start_idx, 'full': (size - start_idx) == buffer.buffer_size,
        'buffer_size': buffer.buffer_size
    }
    with open(filepath, 'wb') as f:
        pickle.dump(partial_data, f)

def load_partial_replay_buffer(buffer, filepath):
    with open(filepath, 'rb') as f:
        partial_data = pickle.load(f)
    buffer.buffer_size = partial_data.get('buffer_size', buffer.buffer_size)
    size = partial_data['size']
    buffer.size = size; buffer.pos = partial_data['pos']; buffer.full = partial_data['full']
    if isinstance(buffer.observations, dict):
        for k in buffer.observations: buffer.observations[k][:size] = partial_data['observations'][k]
    else:
        buffer.observations[:size] = partial_data['observations']
    buffer.actions[:size] = partial_data['actions']
    if isinstance(buffer.next_observations, dict):
        for k in buffer.next_observations: buffer.next_observations[k][:size] = partial_data['next_observations'][k]
    else:
        buffer.next_observations[:size] = partial_data['next_observations']
    buffer.rewards[:size] = partial_data['rewards']; buffer.dones[:size] = partial_data['dones']

# --------------------------------------------------
# کال‌بک
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq; self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -float("inf")
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if not os.path.isdir(self.log_dir) or not any(fname.startswith('monitor') for fname in os.listdir(self.log_dir)):
                return True
            try:
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                if len(y) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0: print(f"\n[Callback] Timesteps: {self.num_timesteps}, Best Reward: {self.best_mean_reward:.2f}, Last 100ep Reward: {mean_reward:.2f}")
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"--> Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)
                        replay_buffer_path = os.path.join(self.log_dir, "replay_buffer.pkl")
                        save_partial_replay_buffer(self.model.replay_buffer, replay_buffer_path)
            except Exception as e:
                print(f"Error in callback: {e}")
        return True

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# --------------------------------------------------
# ساخت محیط
def make_env():
    env = gym.make("PandaReach2-v3", render_mode="rgb_array", reward_type="dense", control_type="joints")
    env = Monitor(env, monitor_path, info_keywords=("is_success",))
    return env

vec_env = DummyVecEnv([make_env])

# --------------------------------------------------
# بارگذاری و ادامه یادگیری
model = SAC.load(best_model, env=vec_env)
replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")
if os.path.exists(replay_buffer_path):
    try:
        load_partial_replay_buffer(model.replay_buffer, replay_buffer_path)
        print("Partial replay buffer loaded successfully.")
    except Exception as e:
        print(f"Could not load replay buffer, starting fresh. Error: {e}")
else:
    print("Replay buffer file not found. Starting with empty buffer.")

model.learn(total_timesteps=total_timesteps_target, reset_num_timesteps=False, log_interval=10, callback=callback)


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
print("\nDone.")

