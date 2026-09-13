"""Webots Controller for BmwX5 - Pure Longitudinal IRL Test (Smoothed Output)"""

import os
import math
import numpy as np
from vehicle import Driver
from collections import deque

# 1. IRL Model Load
MODEL_PATH = "v_table_for_webots.npz"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "v_table_for_webots.npz")

data = np.load(MODEL_PATH)
V = data["V"]
dist_min = float(data["dist_min"])    # -2.0
dist_max = float(data["dist_max"])    # 40.0
speed_min = float(data["speed_min"])  # 0.0
speed_max = float(data["speed_max"])  # 7.0

num_d_bins, num_s_bins = V.shape
d_bin_size = (dist_max - dist_min) / num_d_bins
s_bin_size = (speed_max - speed_min) / num_s_bins
accels = np.linspace(-3.0, 2.0, 11)

# 2. Webots Init
driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())
dt = TIME_STEP / 1000.0

vehicle_node = driver.getFromDef("EGO_VEHICLE")
INTERSECTION_X = -71.3  # 死角交差点中心のX座標

# ハンチング防止用の平滑化バッファ（過去3ステップ分）
acc_history = deque(maxlen=3)

# 3. Control Loop
while driver.step() != -1:
    if vehicle_node is None:
        break

    # 1. 現在状態の取得
    raw_speed = driver.getCurrentSpeed()
    current_speed = 0.0 if (raw_speed is None or math.isnan(raw_speed)) else max(0.0, raw_speed / 3.6)

    pos = vehicle_node.getPosition()
    if pos is None or math.isnan(pos[0]):
        continue
        
    curr_x = pos[0]
    curr_dist = INTERSECTION_X - curr_x  # 正: 手前, 負: 通過後

    # 2. IRLによる最適加速度選択
    d_val = np.nan_to_num((curr_dist - dist_min) / d_bin_size, nan=0.0)
    s_val = np.nan_to_num((current_speed - speed_min) / s_bin_size, nan=0.0)

    d_idx = int(np.clip(d_val, 0, num_d_bins - 1))
    s_idx = int(np.clip(s_val, 0, num_s_bins - 1))

    raw_best_acc = 0.0
    best_val = -float("inf")

    for acc in accels:
        next_speed = np.clip(current_speed + acc * dt, speed_min, speed_max)
        next_dist = curr_dist - current_speed * dt

        next_d_val = np.nan_to_num((next_dist - dist_min) / d_bin_size, nan=0.0)
        next_s_val = np.nan_to_num((next_speed - speed_min) / s_bin_size, nan=0.0)

        next_d_idx = int(np.clip(next_d_val, 0, num_d_bins - 1))
        next_s_idx = int(np.clip(next_s_val, 0, num_s_bins - 1))

        val = V[next_d_idx, next_s_idx]

        if val > best_val:
            best_val = val
            raw_best_acc = acc
        elif abs(val - best_val) < 1e-6:
            if curr_dist > 0 and acc > raw_best_acc:
                raw_best_acc = acc

    # 3. 加速度の移動平均処理（平滑化）
    acc_history.append(raw_best_acc)
    smoothed_acc = float(np.mean(acc_history))

    # 4. アクチュエータ制御
    target_speed = np.clip(current_speed + smoothed_acc * dt, speed_min, speed_max)

    if smoothed_acc < 0:
        driver.setThrottle(0.0)
        driver.setBrakeIntensity(np.clip(abs(smoothed_acc) / 3.0, 0.0, 1.0))
        driver.setCruisingSpeed(0.0)
    else:
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(target_speed * 3.6)

    driver.setSteeringAngle(0.0)

    print(
        f"[IRL Test] Dist: {curr_dist:6.2f}m | Speed: {current_speed:4.2f}m/s"
        f" ({current_speed * 3.6:4.1f}km/h) | RawAccel: {raw_best_acc:4.1f} | SmoothAccel: {smoothed_acc:4.2f}m/s²"
    )