import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# مسیرهای لاگ اولیه و ادامه آموزش
log_dirs = ["/home/moj79/panda_gym2/log/sac_initial"]
continue_dirs = glob.glob("/home/moj79/panda_gym2/log/sac_continue_*")
log_dirs.extend(continue_dirs)

df_list = []
for log_dir in log_dirs:
    csv_paths = glob.glob(os.path.join(log_dir, "**", "monitor*.csv"), recursive=True)
    for path in csv_paths:
        tmp = pd.read_csv(path, comment="#")
        renames = {}
        if "r" in tmp.columns:
            renames["r"] = "reward"
        if "l" in tmp.columns:
            renames["l"] = "length"
        # تغییر نام ستون زمان به "time"
        if "t" in tmp.columns:
            renames["t"] = "time"
        elif "step" in tmp.columns:
            renames["step"] = "time"
        # اگر ستون "time" وجود دارد که نیازی نیست کاری بکنیم
        tmp = tmp.rename(columns=renames)
        df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

# مرتب‌سازی بر اساس ستون موجود
if "time" in df.columns:
    df = df.sort_values(by="time")
elif "length" in df.columns:
    df = df.sort_values(by="length")
else:
    df = df.sort_index()

df.reset_index(drop=True, inplace=True)

# محاسبه میانگین متحرک پاداش با پنجره 100
window = 100
df["mean_reward"] = df["reward"].rolling(window).mean()

# محاسبه درصد موفقیت اگر ستون is_success موجود باشد
if "is_success" in df.columns:
    df["success_rate"] = df["is_success"].rolling(window).mean() * 100
else:
    df["success_rate"] = None

# رسم نمودارها
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(df.index + 1, df["reward"], label="Reward per Episode", linewidth=0.7)
axes[0].plot(df.index + 1, df["mean_reward"], label=f"Mean Reward ({window})", linewidth=1.2)
axes[0].set_title("Reward & Mean Reward")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True)

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

