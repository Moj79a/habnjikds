import gymnasium as gym
import panda_gym
import pickle
from numpngw import write_apng
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

log_dir_sac = "/home/moj79/panda_gym2/log/sac"
"""base_log_dir = "/home/moj79/panda_gym2/log"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # مثل 20230521_103045
log_dir_sac = os.path.join(base_log_dir, f"sac_{run_id}") """
os.makedirs(log_dir_sac, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
monitor_path = os.path.join(log_dir_sac, f"monitor_{run_id}.csv")

def make_env():
    env = gym.make("PandaReach1-v3", render_mode="rgb_array", reward_type="dense", control_type="joints")
    env = Monitor(env, monitor_path, info_keywords=("is_success",))
    return env

vec_env = DummyVecEnv([make_env])


vec_env.reset()



"""done=False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()"""
        
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


check_freq = 10
#log_dir_ddpg = "/home/moj79/panda-gym/log/ddpg"
#log_dir_td3 = "/home/moj79/panda-gym/log/td3"

"""
# ---

log_dir_ddpg = "/home/moj79/panda-gym/log/ddpg"
os.makedirs(log_dir_ddpg, exist_ok=True)

# Logs will be saved in log_dir/monitor.csv
log_env = Monitor(env, log_dir_ddpg)

check_freq = 10
callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir_ddpg)

# ---

total_timesteps = 100000

noise_stddev = 0.2
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_stddev * np.ones(n_actions))


model = DDPG("MultiInputPolicy",
             DummyVecEnv([lambda: log_env]),
             train_freq=(1, "episode"),
             learning_rate=1e-03,
             batch_size=512,
             buffer_size=100000,
             replay_buffer_class=HerReplayBuffer,
             policy_kwargs=dict(net_arch=[128]),
             learning_starts=100,
             tau=0.005,
             gamma=0.99,
             action_noise=action_noise,
             device='auto',
             verbose=2)
model.learn(total_timesteps=total_timesteps, log_interval=50, callback=callback)

"""
# ---
"""
log_dir_td3 = "/home/moj79/panda-gym/log/td3"
os.makedirs(log_dir_td3, exist_ok=True)

# Logs will be saved in log_dir/monitor.csv
log_env = Monitor(env, log_dir_td3)

check_freq = 10
callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir_td3)

# ---

total_timesteps = 100000

noise_stddev = 0.2
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_stddev * np.ones(n_actions))

model = TD3("MultiInputPolicy",
             DummyVecEnv([lambda: log_env]),
             train_freq=(1, "episode"),
             learning_rate=1e-03,
             batch_size=512,
             buffer_size=100000,
             replay_buffer_class=HerReplayBuffer,
             policy_kwargs=dict(net_arch=[128]),
             learning_starts=100,
             tau=0.005,
             gamma=0.99,
             action_noise=action_noise,
             device='auto',
             verbose=2)
model.learn(total_timesteps=total_timesteps, log_interval=50, callback=callback)
"""



# ۲. پیاده‌سازی Monitor روی محیط

#log_env = Monitor(env, log_dir_sac , info_keywords=("is_success",))
#log_env = Monitor(env, monitor_path, info_keywords=("is_success",))
# ۳. کال‌بک برای ذخیره بهترین مدل
check_freq = 10
callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir_sac)

# ۴. تنظیمات کلی آموزش
total_timesteps = 20_000_000

# SAC نیاز به action noise ندارد؛ اگر مایلید حذفش کنید
# noise_stddev = 0.2
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(
#     mean=np.zeros(n_actions),
#     sigma=noise_stddev * np.ones(n_actions)
# )

# ۵. ساخت VecEnv
#vec_env = DummyVecEnv([lambda: log_env])


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
#model.replay_buffer.save(os.path.join(log_dir, "replay_buffer.pkl"))

"""
x, y_ddpg = ts2xy(load_results(log_dir_ddpg), "episodes")
x, y_td3 = ts2xy(load_results(log_dir_td3), "episodes")

mean_rewards_ddpg = []
mean_rewards_td3 = []
n = 0
for i in range(1, 1+ max(len(y_ddpg),len(y_td3))):
    if i % check_freq == 0:
        if i >= len(y_ddpg):
            mean_rewards_ddpg.append(None)
        else:
            mean_reward_ddpg = np.mean(y_ddpg[n*check_freq:(n+1)*check_freq])
            mean_rewards_ddpg.append(mean_reward_ddpg)

        if i >= len(y_td3):
            mean_rewards_td3.append(None)
        else:
            mean_reward_td3 = np.mean(y_td3[n*check_freq:(n+1)*check_freq])
            mean_rewards_td3.append(mean_reward_td3)

        n += 1

episode_counter = np.linspace(check_freq, max(len(y_ddpg),len(y_td3)), int(max(len(y_ddpg),len(y_td3))/check_freq))

plt.plot(episode_counter, mean_rewards_ddpg, label='DDPG')
plt.plot(episode_counter, mean_rewards_td3, label='TD3')

plt.title("Mean Reward Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")

plt.legend()
plt.show()

model_DDPG = DDPG.load("/home/moj79/panda-gym/log/ddpg/best_model.zip", env=env)
model_TD3 = TD3.load("/home/moj79/panda-gym/log/td3/best_model.zip", env=env)

mean_reward, std_reward = evaluate_policy(model_DDPG, env, n_eval_episodes=10)
print("DDPG model:")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

mean_reward, std_reward = evaluate_policy(model_TD3, env, n_eval_episodes=10)

print("TD3 model:")
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
"""
env.close()
