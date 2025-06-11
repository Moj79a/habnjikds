import pybullet as p
import pybullet_data
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Simulation Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# --- 2. Identify Movable Joints ---
num_joints = p.getNumJoints(robotId)
movable_joints_indices = [i for i in range(num_joints) if p.getJointInfo(robotId, i)[2] == p.JOINT_REVOLUTE]
num_movable_joints = len(movable_joints_indices)

# --- Observer Parameters ---
epsilon = 0.01
dt = 1.0 / 240.0
H1 = 2 / epsilon
H2 = 2 / epsilon**2

# --- پارامترهای اخلال ناگهانی ---
disturbance_steps = [3000, 9000 , 9100 , 9101 ,9]  # گام‌هایی که در آنها اخلال اعمال می‌شود
joint_to_disturb_index = 3             # مفصلی که به آن ضربه می‌زنیم
disturbance_torque = 1000         # مقدار گشتاور ضربه (یک عدد بزرگ)

        
# --- Initialize Observer States ---
observer_z1 = [0.0] * num_movable_joints
observer_z2 = [0.0] * num_movable_joints

# --- Data Logging Preparation ---
log_data = []

print("Simulation and data collection started...")
# --- 3. Main Loop ---
try:
    # افزایش تعداد گام‌ها به 15000 برای زمان طولانی‌تر
    for step in range(15000):
        # --- Complex Motion Trajectory for All 7 Joints ---
        target_pos_j0 = 0.7 * math.sin(0.5 * step * dt)
        target_pos_j1 = 0.5 * math.cos(0.8 * step * dt) + 0.2 * math.sin(2.5 * step * dt)
        target_pos_j2 = 0.4 * math.sin(1.5 * step * dt + math.pi/2)
        target_pos_j3 = 0.6 * math.cos(0.4 * step * dt)  # New motion
        target_pos_j4 = 0.6 * math.cos(1.1 * step * dt)
        target_pos_j5 = 0.5 * math.sin(1.3 * step * dt - math.pi/4)  # New motion
        target_pos_j6 = 0.15 * math.cos(3.0 * step * dt)  # New motion
        target_pos_d = 0.55 * math.cos(1.3 * step * dt)
        target_positions = [target_pos_j0, target_pos_j1, target_pos_j2, target_pos_j3, 
target_pos_j4, target_pos_j5, target_pos_j6]


# --- Apply Control Commands ---
        for j in range(num_movable_joints):
            # ==============================================================================
            # --- بخش جدید: اعمال اخلال در گام‌های مشخص ---
            # ==============================================================================
            # اگر گام فعلی، گام اخلال بود و مفصل فعلی، مفصل مورد نظر ما بود
            if (step >= 7900 and step <= 8900) or \
   (step >= 9000 and step <= 12000):
                # یک گشتاور ناگهانی اعمال کن
                p.setJointMotorControl2(robotId, movable_joints_indices[j],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=0.05+target_pos_d+target_positions[j])
            else:
                # در غیر این صورت، حرکت عادی را ادامه بده
                p.setJointMotorControl2(robotId, movable_joints_indices[j],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=target_positions[j])
            # ==============================================================================


        p.stepSimulation()

        time.sleep(dt)
        
        
        # --- Read States and Update Observer ---
        joint_states = p.getJointStates(robotId, movable_joints_indices)
        current_joint_angles = [s[0] for s in joint_states]
        current_joint_velocities = [s[1] for s in joint_states]

        for j in range(num_movable_joints):
            y_measured = current_joint_angles[j]
            z1, z2 = observer_z1[j], observer_z2[j]
            error = y_measured - z1
            z1_dot = z2 + H1 * error
            z2_dot = H2 * error
            observer_z1[j] += z1_dot * dt
            observer_z2[j] += z2_dot * dt
        
        # --- Log Data for this Step ---
        step_log = {'time': step * dt}
        for j in range(num_movable_joints):
            step_log[f'pos_true_j{j}'] = current_joint_angles[j]
            # ... (rest of the logging code remains the same)
            step_log[f'pos_est_j{j}'] = observer_z1[j]
            step_log[f'pos_error_j{j}'] = current_joint_angles[j] - observer_z1[j]
            step_log[f'vel_true_j{j}'] = current_joint_velocities[j]
            step_log[f'vel_est_j{j}'] = observer_z2[j]
            step_log[f'vel_error_j{j}'] = current_joint_velocities[j] - observer_z2[j]
        log_data.append(step_log)
        
except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    p.disconnect()
    print("Simulation finished.")


# --- Data Processing, Metrics Calculation, and Plotting ---
print("Processing data and saving results...")

# Convert data list to a pandas DataFrame for easy analysis
df = pd.DataFrame(log_data)

# --- Final Metrics Calculation (for joint 0 as an example) ---
# This part can remain as is for a quick console summary
joint_to_analyze = 0
pos_error_col = f'pos_error_j{joint_to_analyze}'
vel_error_col = f'vel_error_j{joint_to_analyze}'
ise_pos = (df[pos_error_col]**2).sum() * dt
ise_vel = (df[vel_error_col]**2).sum() * dt
mae_pos = df[pos_error_col].abs().mean()
mae_vel = df[vel_error_col].abs().mean()
rmse_pos = (df[pos_error_col]**2).mean()**0.5
rmse_vel = (df[vel_error_col]**2).mean()**0.5

print(f"\n--- Final Analysis Summary for Joint {joint_to_analyze} (as an example) ---")
print(f"ISE (Position): {ise_pos:.6f} | MAE (Position): {mae_pos:.6f} | RMSE (Position): {rmse_pos:.6f}")
print(f"ISE (Velocity): {ise_vel:.6f} | MAE (Velocity): {mae_vel:.6f} | RMSE (Velocity): {rmse_vel:.6f}")

# --- Save all data to a CSV file ---
csv_filename = 'observer_performance_data.csv'
df.to_csv(csv_filename, index=False)
print(f"\nAll data saved to '{csv_filename}'.")

# ==============================================================================
# --- بخش جدید و کامل رسم نمودار برای تمام مفاصل ---
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- نمودار ۱: مقایسه سرعت‌ها برای تمام ۷ مفصل در ساب‌پلات‌ها ---
fig_vel, axes_vel = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
axes_vel = axes_vel.flatten() # تبدیل ماتریس 4x2 به یک لیست برای دسترسی آسان
for j in range(num_movable_joints):
    ax = axes_vel[j]
    ax.plot(df['time'], df[f'vel_true_j{j}'], label='True Velocity', linewidth=2)
    ax.plot(df['time'], df[f'vel_est_j{j}'], label='Estimated Velocity', linestyle='--', linewidth=2)
    ax.set_title(f'Joint {j} Velocity')
    ax.set_ylabel('Velocity (rad/s)')
    ax.legend()
# حذف ساب‌پلات خالی آخر
if len(axes_vel) > num_movable_joints:
    fig_vel.delaxes(axes_vel[-1])

fig_vel.suptitle('Velocity Comparison for All Joints', fontsize=16)
fig_vel.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('velocity_comparison_all_joints.png')
print("Velocity comparison plot for all joints saved to 'velocity_comparison_all_joints.png'.")


# --- نمودار ۲: مقایسه موقعیت‌ها برای تمام ۷ مفصل در ساب‌پلات‌ها ---
fig_pos, axes_pos = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
axes_pos = axes_pos.flatten()
for j in range(num_movable_joints):
    ax = axes_pos[j]
    ax.plot(df['time'], df[f'pos_true_j{j}'], label='True Position', linewidth=2)
    ax.plot(df['time'], df[f'pos_est_j{j}'], label='Estimated Position', linestyle='--', linewidth=2)
    ax.set_title(f'Joint {j} Position')
    ax.set_ylabel('Position (rad)')
    ax.legend()
# حذف ساب‌پلات خالی آخر
if len(axes_pos) > num_movable_joints:
    fig_pos.delaxes(axes_pos[-1])

fig_pos.suptitle('Position Comparison for All Joints', fontsize=16)
fig_pos.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('position_comparison_all_joints.png')
print("Position comparison plot for all joints saved to 'position_comparison_all_joints.png'.")


# --- نمودار ۳: همگرایی معیارهای خطا (برای سرعت) - تمام مفاصل روی یک نمودار ---
fig_err, axes_err = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

# ساب‌پلات ISE
for j in range(num_movable_joints):
    ise_cumulative = (df[f'vel_error_j{j}']**2).cumsum() * dt
    axes_err[0].plot(df['time'], ise_cumulative, label=f'Joint {j}')
axes_err[0].set_title('Integral Squared Error (ISE) - Velocity')
axes_err[0].set_ylabel('ISE')
axes_err[0].legend()

# ساب‌پلات MAE
for j in range(num_movable_joints):
    mae_cumulative = df[f'vel_error_j{j}'].abs().expanding().mean()
    axes_err[1].plot(df['time'], mae_cumulative, label=f'Joint {j}')
axes_err[1].set_title('Mean Absolute Error (MAE) - Velocity')
axes_err[1].set_ylabel('MAE')
axes_err[1].legend()

# ساب‌پلات RMSE
for j in range(num_movable_joints):
    rmse_cumulative = ((df[f'vel_error_j{j}']**2).expanding().mean())**0.5
    axes_err[2].plot(df['time'], rmse_cumulative, label=f'Joint {j}')
axes_err[2].set_title('Root Mean Squared Error (RMSE) - Velocity')
axes_err[2].set_ylabel('RMSE')
axes_err[2].legend()

axes_err[2].set_xlabel('Time (s)')
fig_err.suptitle('Convergence of Velocity Error Metrics for All Joints', fontsize=16)
fig_err.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('error_metrics_all_joints.png')
print("Error metrics convergence plot for all joints saved to 'error_metrics_all_joints.png'.")

plt.show()
