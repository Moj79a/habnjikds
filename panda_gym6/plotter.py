import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
log_filepath = "observer_log_cumulative.csv" # تعریف مسیر فایل لاگ رویتگر

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
#df = df[df['episode'] >= 300000]

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

