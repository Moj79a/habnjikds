import sys
import gymnasium as gym
import panda_gym


env = gym.make('PandaReachOri-v3', render_mode="human", reward_type="sparse", control_type="ee")
obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # اقدام تصادفی
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
