import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# مسیر پوشه‌ی لاگ‌های SAC
#log_dir = "/home/moj79/panda-gym/log/sac"
log_dir = "/home/moj79/panda_gym2/log/sac"

# پیدا کردن همه‌ی فایل‌های monitor*.csv
csv_paths = glob.glob(os.path.join(log_dir, "**", "monitor*.csv"), recursive=True)
csv_paths = sorted(csv_paths, key=lambda p: os.path.basename(p))
if not csv_paths:
    raise FileNotFoundError(f"No monitor*.csv found under {log_dir}.")

# خواندن و ادغام همه‌ی فایل‌ها
df_list = []
episode_offset = 0  # برای شماره‌گذاری تجمعی اپیزودها
for path in csv_paths:
    tmp = pd.read_csv(path, comment="#")
    renames = {}
    if "r" in tmp.columns: renames["r"] = "reward"
    if "l" in tmp.columns: renames["l"] = "length"
    if "t" in tmp.columns: renames["t"] = "time"
    tmp = tmp.rename(columns=renames)
    tmp["episode"] = tmp.index + 1 + episode_offset
    episode_offset += len(tmp)
    
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)
df["episode"] = df.index + 1
df = df.sort_values("episode").reset_index(drop=True)



# محاسبه میانگین متحرک پاداش با پنجره 100
window = 100
df["mean_reward"] = df["reward"].rolling(window).mean()
df["mean_length"]  = df["length"].rolling(window).mean()
df["success_rate"] = df["is_success"].rolling(window).mean() * 100 \
                     if "is_success" in df.columns else None


# ایجاد ۲ subplot کنار هم
fig, axes = plt.subplots(2, 2, figsize=(12, 8))


# 1) پاداش در هر اپیزود
axes[0, 0].plot(df["episode"], df["reward"], linewidth=0.7)
axes[0, 0].set_title("Reward per Episode")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].grid(True)

# 2) میانگین پاداش (پنجره‌ی ۱۰۰)
axes[0, 1].plot(df["episode"], df["mean_reward"], linewidth=1.2)
axes[0, 1].set_title(f"Mean Reward (Window={window})")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Mean Reward")
axes[0, 1].grid(True)

# 3) طول هر اپیزود
if "length" in df.columns:
    axes[1, 0].plot(df["episode"], df["mean_length"], linewidth=1.2)
    axes[1, 0].set_title(f"Mean Episode Length (Window={window})")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean Length")
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(0.5, 0.5, "No 'length' data", ha='center', va='center')
    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

# 4) نرخ موفقیت
if "success_rate" in df.columns:
    axes[1, 1].plot(df["episode"], df["success_rate"], linewidth=1.2)
    axes[1, 1].set_title(f"Success Rate (Window={window})")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Success Rate (%)")
    axes[1, 1].grid(True)
else:
    axes[1, 1].text(0.5, 0.5, "No 'is_success' data", ha='center', va='center')
    axes[1, 1].set_title("Success Rate (%)")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

plt.tight_layout()
plt.show()

