import gymnasium as gym
import panda_gym
from stable_baselines3 import sac
import time

# ایجاد محیط
env = gym.make("PandaReachOri-v3", render_mode="human")

# بارگذاری مدل
model = env.load("/home/moj79/panda-gym/log/sac/best_model.zip", env=env)

# تست مدل با تأخیر
obs, _ = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)  # تأخیر 50 میلی‌ثانیه بین هر مرحله
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
