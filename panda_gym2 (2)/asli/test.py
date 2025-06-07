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
for path in csv_paths:
    tmp = pd.read_csv(path, comment="#")
    renames = {}
    if "r" in tmp.columns: renames["r"] = "reward"
    if "l" in tmp.columns: renames["l"] = "length"
    if "t" in tmp.columns: renames["t"] = "time"
    tmp = tmp.rename(columns=renames)
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)

# محاسبه میانگین متحرک پاداش با پنجره 100
window = 100
df["mean_reward"] = df["reward"].rolling(window).mean()

# محاسبه درصد موفقیت (اگر ستون is_success موجود باشد)
if "is_success" in df.columns:
    df["success_rate"] = df["is_success"].rolling(window).mean() * 100
else:
    df["success_rate"] = None

# ایجاد ۲ subplot کنار هم
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# subplot اول: پاداش هر اپیزود و میانگین متحرک
axes[0].plot(df.index + 1, df["reward"], label="Reward per Episode", linewidth=0.7)
axes[0].plot(df.index + 1, df["mean_reward"], label=f"Mean Reward ({window})", linewidth=1.2)
axes[0].set_title("Reward & Mean Reward")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True)

# subplot دوم: درصد موفقیت
if df["success_rate"].notnull().any():
    axes[1].plot(df.index + 1, df["success_rate"], color="green", linewidth=1.2)
    axes[1].set_title(f"Success Rate (%) (Window={window})")
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

