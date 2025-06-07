import os
import gymnasium as gym
import panda_gym
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from datetime import datetime

# --------------------------------------------------
# ۱) مسیرها و تنظیمات
log_dir       = "/home/moj79/panda_gym2/log/sac"
best_model    = os.path.join(log_dir, "best_model.zip")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
monitor_path = os.path.join(log_dir, f"monitor_{run_id}.csv")
total_timesteps_target = 6_000_000  # تعداد گام نهایی مورد نظر

# --------------------------------------------------
# ۲) کال‌بک ذخیره بهترین مدل با متدهای کامل
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir    = log_dir
        self.save_path  = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -float("inf")
        self.episode_count    = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # این متد باید حتماً پیاده‌سازی شود
        return True

    def _on_rollout_end(self) -> bool:
        # این متد بعد از هر rollout فراخوانی می‌شود
        self.episode_count += 1
        if self.episode_count % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "episodes")
            if len(x) > 0:
                mean_reward = y[-self.check_freq :].mean()
                if self.verbose > 0:
                    print(f"[Callback] Num episodes: {self.episode_count} | "
                          f"Last mean reward: {mean_reward:.2f} | "
                          f"Best mean reward: {self.best_mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"[Callback] Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    with open(os.path.join(self.log_dir, "replay_buffer.pkl"), "wb") as f:
                        save_partial_replay_buffer(self.model.replay_buffer, os.path.join(self.log_dir, "replay_buffer.pkl"), max_size=100000)
        return True


callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)


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

        
def load_partial_replay_buffer(buffer, filepath):
    import pickle
    with open(filepath, 'rb') as f:
        partial_data = pickle.load(f)

    buffer_size = partial_data.get('buffer_size', buffer.buffer_size)
    buffer.buffer_size = buffer_size

    size = partial_data['size']
    buffer.size = size
    buffer.pos = partial_data['pos']
    buffer.full = partial_data['full']

    if isinstance(buffer.observations, dict):
        for key in buffer.observations.keys():
            buffer.observations[key][:size] = partial_data['observations'][key]
    else:
        buffer.observations[:size] = partial_data['observations']

    buffer.actions[:size] = partial_data['actions']

    if isinstance(buffer.next_observations, dict):
        for key in buffer.next_observations.keys():
            buffer.next_observations[key][:size] = partial_data['next_observations'][key]
    else:
        buffer.next_observations[:size] = partial_data['next_observations']

    buffer.rewards[:size] = partial_data['rewards']
    buffer.dones[:size] = partial_data['dones']



# --------------------------------------------------
# ۳) ساخت محیط دقیقاً مثل آموزش با Monitor
def make_env():
    env = gym.make(
        "PandaReach2-v3",
        render_mode="rgb_array",
        reward_type="dense",
        control_type="joints"
    )
    env = Monitor(env, monitor_path, info_keywords=("is_success",))
    return env

# ۴) ساخت VecEnv (در صورت نیاز می‌توانید VecNormalize هم اضافه کنید)
vec_env = DummyVecEnv([make_env])
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# --------------------------------------------------
# ۵) بارگذاری مدل از فایل best_model.zip
model = SAC.load(best_model, env=vec_env)

replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

if os.path.exists(replay_buffer_path):
    load_partial_replay_buffer(model.replay_buffer, replay_buffer_path)
    print("Partial replay buffer loaded successfully.")
else:
    print("Replay buffer file not found. Starting with empty buffer.")

#model.replay_buffer.load(os.path.join(log_dir, "replay_buffer.pkl"))

# --------------------------------------------------
# ۶) ادامه‌ی یادگیری تا رسیدن به total_timesteps_target
model.learn(
    total_timesteps=total_timesteps_target,
    reset_num_timesteps=False,  # ادامه‌ی شماره‌گذاری گام‌ها
    log_interval=50,
    callback=callback
)
# ذخیره‌ی مدل نهایی
model.save(os.path.join(log_dir, "final_model"))
# ذخیره‌ی بافر نهایی (اگر لازم داری)
with open(os.path.join(log_dir, "final_replay_buffer.pkl"), "wb") as f:
    pickle.dump(model.replay_buffer, f)


# --------------------------------------------------
# ۷) بستن محیط
model.env.close()

