import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# مسیر پوشه‌ی لاگ‌های SAC
#log_dir = "/home/moj79/panda-gym/log/sac"
log_dir = "/home/moj79/panda_gym2/log/sac"

# پیدا کردن همه‌ی فایل‌های monitor*.csv
csv_paths = glob.glob(os.path.join(log_dir, "**", "monitor*.csv"), recursive=True)
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
    episode_offset += len(tmp)
    
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)
df["episode"] = df.index + 1
if "time" in df.columns:
    df = df.sort_values("time").reset_index(drop=True)


# محاسبه میانگین متحرک پاداش با پنجره 100
window = 100
df["mean_reward"] = df["reward"].rolling(window).mean()
df["success_rate"] = df["is_success"].rolling(window).mean() * 100 \
                     if "is_success" in df.columns else None


# ایجاد ۲ subplot کنار هم
fig, axes = plt.subplots(1, 2, figsize=(12, 4))


# نمودار پاداش بر حسب اپیزود تجمعی
axes[0].plot(df["episode"], df["reward"], label="Reward per Episode", linewidth=0.7)
axes[0].plot(df["episode"], df["mean_reward"], label=f"Mean Reward ({window})", linewidth=1.2)
axes[0].set_title("Reward & Mean Reward")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True)

# نمودار نرخ موفقیت بر حسب اپیزود تجمعی
if df["success_rate"].notnull().any():
    axes[1].plot(df["episode"], df["success_rate"], label="Success Rate (%)", linewidth=1.2)
    axes[1].set_title(f"Success Rate (Window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].grid(True)
else:
    axes[1].text(0.5, 0.5, "No 'is_success' data", ha='center', va='center')
    axes[1].set_title("Success Rate (%)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])


plt.tight_layout()
plt.show()

