"""
================================================================================
 waypoint_recorder_controller.py
 手動運転によるウェイポイント記録用コントローラ

【目的】
  webots_yolo_bridge_controller.py が使う waypoints.json を、実際に手動で
  右車線・カーブに沿って走らせながら自動生成する。座標を目視で調べて手打ちする
  手間を省くための開発工数削減ツール。

【操作方法】
  Webotsの3Dビューをクリックしてアクティブにしてから操作する:
    ↑ : 加速
    ↓ : 減速
    ← / → : ステアリング
  一定時間ごとに自動でGPS位置をwaypoints.jsonへ追記保存し続けるので、
  意図した経路を1周(または区間の走行)し終えたらシミュレーションを一時停止すればよい。

【配置方法】
  <Webotsプロジェクト>/controllers/waypoint_recorder_controller/waypoint_recorder_controller.py

  city.wbtのPriusノードの controller フィールドを一時的に
  "waypoint_recorder_controller" に変更して実行する。
  記録が終わったら、生成された waypoints.json を
  controllers/webots_yolo_bridge_controller/ フォルダにコピーし、
  Priusのcontrollerフィールドを "webots_yolo_bridge_controller" に戻すこと。

【出力】
  このコントローラのフォルダ内に waypoints.json が生成される。
  形式: {"waypoints": [[x1, z1], [x2, z2], ...]}
================================================================================
"""

import json
import os

from controller import Keyboard
from vehicle import Driver


# ==============================================================================
# 0. CONFIG
# ==============================================================================
STEERING_STEP = 0.02
MAX_STEERING = 0.5
SPEED_STEP_KMH = 2.0
MAX_SPEED_KMH = 25.0
WAYPOINT_RECORD_INTERVAL_SEC = 0.5   # この間隔でGPS位置を記録する
OUTPUT_PATH = "waypoints.json"       # このコントローラのフォルダ内に生成される
GPS_DEVICE_NAME = "gps"


# ==============================================================================
# 1. 初期化
# ==============================================================================
driver = Driver()
timestep = int(driver.getBasicTimeStep())

keyboard = driver.getKeyboard()
keyboard.enable(timestep)

gps = driver.getDevice(GPS_DEVICE_NAME)
gps.enable(timestep)

steering_angle = 0.0
speed_kmh = 0.0
waypoints = []
last_record_time = 0.0

driver.setSteeringAngle(0.0)
driver.setCruisingSpeed(0.0)

print("=" * 70)
print("[INFO] 手動運転によるウェイポイント記録を開始します。")
print("[INFO] Webotsの3Dビューをクリックしてアクティブにしてから操作してください。")
print("[INFO]   ↑:加速  ↓:減速  ←/→:ステアリング")
print(f"[INFO] {WAYPOINT_RECORD_INTERVAL_SEC}秒ごとに自動で {OUTPUT_PATH} へ保存されます。")
print("[INFO] 経路の記録が終わったらシミュレーションを一時停止してください。")
print("=" * 70)


def save_waypoints():
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"waypoints": waypoints}, f, indent=2)


# ==============================================================================
# 2. メインループ (キーを離したらハンドルが戻る修正版)
# ==============================================================================
while driver.step() != -1:
    key = keyboard.getKey()
    
    # 毎ステップ、キー入力フラグをチェック
    steering_input = False
    
    while key != -1:
        if key == Keyboard.UP:
            speed_kmh = min(MAX_SPEED_KMH, speed_kmh + SPEED_STEP_KMH)
        elif key == Keyboard.DOWN:
            speed_kmh = max(0.0, speed_kmh - SPEED_STEP_KMH)
        elif key == Keyboard.LEFT:
            steering_angle = max(-MAX_STEERING, steering_angle - STEERING_STEP)
            steering_input = True
        elif key == Keyboard.RIGHT:
            steering_angle = min(MAX_STEERING, steering_angle + STEERING_STEP)
            steering_input = True
        
        key = keyboard.getKey()

    # 左右のキーが押されていない場合は、ハンドルを徐々にまっすぐに戻す（センタリング）
    if not steering_input:
        if steering_angle > 0:
            steering_angle = max(0.0, steering_angle - STEERING_STEP * 2)
        elif steering_angle < 0:
            steering_angle = min(0.0, steering_angle + STEERING_STEP * 2)

    driver.setSteeringAngle(steering_angle)
    driver.setCruisingSpeed(speed_kmh)

    sim_time = driver.getTime()
    if sim_time - last_record_time >= WAYPOINT_RECORD_INTERVAL_SEC:
        pos = gps.getValues()  # [X, Y, Z]

        # 【修正】高さ(pos[2]) ではなく、平面移動である pos[1] (Y軸) を保存する
        waypoints.append([round(pos[0], 3), round(pos[1], 3)])

        last_record_time = sim_time
        save_waypoints()  # 記録の都度、上書き保存
        print(
            f"[REC t={sim_time:5.1f}s] waypoint追加: "
            f"({pos[0]:.2f}, {pos[1]:.2f})  合計{len(waypoints)}点  "
            f"speed={speed_kmh:.0f}km/h steering={steering_angle:.2f}"
        )