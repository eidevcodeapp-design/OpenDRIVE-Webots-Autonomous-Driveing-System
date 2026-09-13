"""
================================================================================
 data_collector.py
 死角交差点における「手動走行データ収集」用 Ego車両コントローラ(IRL学習用)

【役割】
  自動運転(webots_yolo_bridge_controller.py)の代わりに、テイク撮り用に
  Ego車両のcontrollerとして一時的に差し替えて使うスクリプト。
  人間がキーボードで運転しながら、
    - 自車の位置・速度
    - 死角交差点までの距離
    - 歩行者との距離
    - 人間の操作量(throttle/brake/steering)
  を1ステップごとにCSVへ記録する。あわせて、死角交差点への接近を検知すると
  歩行者(飛び出しイベント)も自前で再現するため、実際の評価シナリオと
  同じ状況に対する人間の減速・回避判断を収集できる。

【既存スクリプトとの関係(競合回避)】
  Webotsは1つのRobotノードに1つのcontrollerしか設定できないため、
  「自動運転で走らせる」時と「手動でデータ収集する」時とで、
  Ego車両ノードの controller フィールドを
    webots_yolo_bridge_controller  <->  data_collector
  に手動で切り替えて使う想定(同時には動かさない)。

  歩行者の死角飛び出しロジックは、データ収集時はこのスクリプト自身が
  Supervisor権限で担当する(blind_spot_pedestrian_supervisor.py と同じ処理を内包)。
  そのため、データ収集セッション中は blind_spot_pedestrian_supervisor.py を
  紐付けたRobotノードを無効化(削除、またはcontrollerを<none>に)しておくこと。
  2つのSupervisorが同時に同じ歩行者ノードを動かすと、位置更新がお互いに
  上書きし合ってガクつく可能性があるため。
  自動運転での評価走行に戻す際は、逆に data_collector を外して
  webots_yolo_bridge_controller.py + blind_spot_pedestrian_supervisor.py の
  組み合わせに戻せばよい。

【重要: Ego車両ノードにも supervisor フィールドをTRUEにすること】
  このスクリプトは「手動運転(Driverクラス)」と「歩行者/自車位置の強制リセット
  (Supervisor権限)」の両方を1つのプロセスで行う。Webots R2023a以降であれば、
  Car/Driver系PROTO(Priusなど)の supervisor フィールドをTRUEにするだけで、
  同じcontroller内でSupervisor専用API(getSelf, getFromDef, getField等)が
  使えるようになる。R2022b以前ではこの機能が働かないため、その場合は
  「Rキーでの即時リセット」は使えず、Ctrl+Shift+R(ワールドRevert)での
  リセットのみになる(録画の開始/停止(Sキー)や走行データの記録自体は
  supervisor権限が無くても問題なく動作する)。

【操作方法】
  ↑ : アクセル(押している間だけ加速、離すと自然に戻る)
  ↓ : ブレーキ(押している間だけ制動)
  ← / → : ステア(押している間だけ切れる。離すと中央へ戻る)
  S キー: 記録(REC)の開始/停止をトグル。開始のたびに新しいCSVファイルを作成する。
  R キー: 自車・歩行者を初期位置に即時リセットし、シナリオを再アーム。
          記録中だった場合は、そのテイクは録り直しとして自動的に記録停止される。

【配置方法】
  <Webotsプロジェクト>/controllers/data_collector/data_collector.py

【出力】
  <Webotsプロジェクト>/controllers/data_collector/irl_logs/ 以下に
  take_0001_20260904_153012.csv のようなファイルが1テイクごとに作られる。
================================================================================
"""

import csv
import math
import os
from datetime import datetime

from controller import Keyboard
from vehicle import Driver


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class Config:
    # --- ノードのDEF名(既存のSupervisorスクリプトと同じ名前にすること) ---
    EGO_DEF_NAME = "EGO_VEHICLE"
    PEDESTRIAN_DEF_NAME = "PEDESTRIAN_NPC"

    GPS_DEVICE_NAME = "gps"

    # --- 死角交差点(既存スクリプトと同じ値) ---
    INTERSECTION_POSITION_XZ = [-96.23, 0.29]
    PEDESTRIAN_TRIGGER_DISTANCE_M = 15.0

    # --- 初期位置(Scene Treeで一度手動配置し、表示された値をコピーして置き換えること) ---
    EGO_INITIAL_TRANSLATION = [-97.02, -52.45, 0.317]     # [x, y, z] 例。交差点の手前、助走区間の開始点
    EGO_INITIAL_ROTATION = [0.0, 0.0, 1.0, 1.57]       # [x, y, z, angle(rad)] 例。進行方向に合わせる

    PEDESTRIAN_INITIAL_TRANSLATION = [-96.0, 0.9, 5.0]
    PEDESTRIAN_INITIAL_ROTATION = [0.0, 1.0, 0.0, 1.57]

    PEDESTRIAN_WALK_DIRECTION_XZ = [0.0, 1.0]  # 道路の向きに合わせて調整
    PEDESTRIAN_WALK_SPEED_MPS = 1.8
    PEDESTRIAN_WALK_DISTANCE_M = 6.0

    # --- 手動操作の感度 ---
    STEERING_STEP_RAD_PER_STEP = 0.008      # 左右キーを押している間の1ステップあたりの舵角変化
    STEERING_RETURN_RATE = 0.03            # 何も押していない時に中央へ戻る速さ(1ステップあたり)
    STEERING_MAX_RAD = 0.5

    THROTTLE_STEP_PER_STEP = 0.04          # 上キーを押している間のアクセル踏み込み速さ
    THROTTLE_RELEASE_RATE = 0.06           # 離した時の戻り速さ
    BRAKE_STEP_PER_STEP = 0.08             # 下キーを押している間のブレーキ踏み込み速さ
    BRAKE_RELEASE_RATE = 0.10              # 離した時の戻り速さ

    # --- ログ出力 ---
    LOG_DIR = "irl_logs"


# ==============================================================================
# 1. 初期化
# ==============================================================================
driver = Driver()
timestep = int(driver.getBasicTimeStep())
driver.setGear(1)  # 追加：ギアを1速に入れる

gps = driver.getDevice(Config.GPS_DEVICE_NAME)
gps.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

os.makedirs(Config.LOG_DIR, exist_ok=True)

print("[INFO] data_collector.py を起動しました。")
print("[INFO] 操作: 矢印キー(加速/減速/ステア), S=記録開始・停止, R=即時リセット")


# --- Supervisor権限の取得を試みる(自車・歩行者の位置リセット用) ---
supervisor_ready = False
ego_self_node = None
ego_translation_field = None
ego_rotation_field = None
pedestrian_node = None
pedestrian_translation_field = None
pedestrian_rotation_field = None

try:
    ego_self_node = driver.getSelf()
    pedestrian_node = driver.getFromDef(Config.PEDESTRIAN_DEF_NAME)
    if ego_self_node is None:
        raise RuntimeError("driver.getSelf() が None を返しました。")
    if pedestrian_node is None:
        raise RuntimeError(
            f"DEF '{Config.PEDESTRIAN_DEF_NAME}' の歩行者ノードが見つかりません。"
        )
    ego_translation_field = ego_self_node.getField("translation")
    ego_rotation_field = ego_self_node.getField("rotation")
    pedestrian_translation_field = pedestrian_node.getField("translation")
    pedestrian_rotation_field = pedestrian_node.getField("rotation")
    supervisor_ready = True
    print("[INFO] Supervisor権限を取得しました。Rキーでの即時リセットが使えます。")
except Exception as e:
    print("=" * 70)
    print(f"[WARN] Supervisor機能が使えませんでした({e})。")
    print("[WARN] Rキーでの即時リセット・歩行者の飛び出し再現は無効化されます。")
    print("[WARN] 走行データの記録自体は問題なく行えます。")
    print("[WARN] 有効化するには: 1) Webots R2023a以降を使う")
    print("[WARN]                2) Ego車両PROTOノードの 'supervisor' フィールドをTRUEにする")
    print("=" * 70)


# ==============================================================================
# 2. ユーティリティ
# ==============================================================================
def distance_2d(p1_xz, p2_xz) -> float:
    return math.hypot(p1_xz[0] - p2_xz[0], p1_xz[1] - p2_xz[1])


def reset_scenario():
    """自車・歩行者を初期位置へ即時リセットし、シナリオ状態を再アームする。"""
    global pedestrian_triggered, pedestrian_walked_distance_m
    global steering_angle, throttle, brake

    if supervisor_ready:
        ego_translation_field.setSFVec3f(Config.EGO_INITIAL_TRANSLATION)
        ego_rotation_field.setSFRotation(Config.EGO_INITIAL_ROTATION)
        ego_self_node.resetPhysics()

        pedestrian_translation_field.setSFVec3f(Config.PEDESTRIAN_INITIAL_TRANSLATION)
        pedestrian_rotation_field.setSFRotation(Config.PEDESTRIAN_INITIAL_ROTATION)
        pedestrian_node.resetPhysics()

    pedestrian_triggered = False
    pedestrian_walked_distance_m = 0.0
    steering_angle = 0.0
    throttle = 0.0
    brake = 0.0
    driver.setSteeringAngle(0.0)
    driver.setThrottle(0.0)
    driver.setBrakeIntensity(0.0)

    if recording:
        stop_recording(aborted=True)

    print("[INFO] シナリオをリセットしました。次のテイクを開始できます。")


# ==============================================================================
# 3. CSV記録
# ==============================================================================
take_count = 0
recording = False
csv_file = None
csv_writer = None

CSV_HEADER = [
    "timestamp_sec",
    "ego_x", "ego_y", "ego_z",
    "ego_speed_mps",
    "distance_to_intersection_m",
    "distance_to_pedestrian_m",
    "human_throttle",
    "human_brake",
    "human_steering_rad",
]


def start_recording():
    global take_count, recording, csv_file, csv_writer
    take_count += 1
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(Config.LOG_DIR, f"take_{take_count:04d}_{ts_str}.csv")
    csv_file = open(filename, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)
    recording = True
    print(f"[REC] 記録を開始しました: {filename}")


def stop_recording(aborted: bool = False):
    global recording, csv_file, csv_writer
    if csv_file is not None:
        csv_file.close()
    recording = False
    csv_file = None
    csv_writer = None
    if aborted:
        print("[REC] リセットのため、このテイクの記録を停止しました(録り直し)。")
    else:
        print("[REC] 記録を停止しました。")


# ==============================================================================
# 4. メインループ用の状態変数
# ==============================================================================
steering_angle = 0.0
throttle = 0.0
brake = 0.0

pedestrian_triggered = False
pedestrian_walked_distance_m = 0.0

_prev_gps_pos_for_speed = None
_prev_gps_time_for_speed = None

_prev_pressed_keys = set()

reset_scenario()  # 起動時(ワールド開始/Revert時)に必ず初期位置へ


def update_ego_speed_mps(current_pos_xz, current_time) -> float:
    global _prev_gps_pos_for_speed, _prev_gps_time_for_speed
    if _prev_gps_pos_for_speed is None:
        _prev_gps_pos_for_speed = current_pos_xz
        _prev_gps_time_for_speed = current_time
        return 0.0
    dt = current_time - _prev_gps_time_for_speed
    if dt <= 1e-6:
        return 0.0
    dist = distance_2d(current_pos_xz, _prev_gps_pos_for_speed)
    speed = dist / dt
    _prev_gps_pos_for_speed = current_pos_xz
    _prev_gps_time_for_speed = current_time
    return speed


# ==============================================================================
# 5. メインループ
# ==============================================================================
frame_index = 0

while driver.step() != -1:
    sim_time_sec = driver.getTime()

    # --- キーボード読み取り(1ステップで複数キーが同時に押される可能性があるため
    #     -1が返るまで繰り返し取得する) ---
    pressed_keys = set()
    key = keyboard.getKey()
    while key != -1:
        pressed_keys.add(key)
        key = keyboard.getKey()

    # 'R' / 'S' はトグsル動作なので、押した瞬間(立ち上がりエッジ)だけ反応させる
    r_pressed_now = ord("R") in pressed_keys and ord("R") not in _prev_pressed_keys
    s_pressed_now = ord("S") in pressed_keys and ord("S") not in _prev_pressed_keys
    _prev_pressed_keys = pressed_keys

    if r_pressed_now:
        reset_scenario()

    if s_pressed_now:
        if recording:
            stop_recording(aborted=False)
        else:
            start_recording()

    # --- 手動操作: アクセル/ブレーキ/ステアリング ---
    # キー状態の判定（定数判定の漏れを防止）
    is_up = Keyboard.UP in pressed_keys
    is_down = Keyboard.DOWN in pressed_keys
    is_left = Keyboard.LEFT in pressed_keys
    is_right = Keyboard.RIGHT in pressed_keys

    # アクセル & ブレーキの更新
    if is_up:
        throttle = min(1.0, throttle + Config.THROTTLE_STEP_PER_STEP)
        brake = max(0.0, brake - Config.BRAKE_RELEASE_RATE)
    else:
        throttle = max(0.0, throttle - Config.THROTTLE_RELEASE_RATE)

    if is_down:
        brake = min(1.0, brake + Config.BRAKE_STEP_PER_STEP)
        throttle = max(0.0, throttle - Config.THROTTLE_RELEASE_RATE)
    else:
        brake = max(0.0, brake - Config.BRAKE_RELEASE_RATE)

    # ステアリングの更新
    if is_left:
        steering_angle -= Config.STEERING_STEP_RAD_PER_STEP
    elif is_right:
        steering_angle += Config.STEERING_STEP_RAD_PER_STEP
    else:
        if steering_angle > 0:
            steering_angle = max(0.0, steering_angle - Config.STEERING_RETURN_RATE)
        elif steering_angle < 0:
            steering_angle = min(0.0, steering_angle + Config.STEERING_RETURN_RATE)

    steering_angle = max(-Config.STEERING_MAX_RAD, min(Config.STEERING_MAX_RAD, steering_angle))

    # --- 車両への出力設定 ---
    driver.setSteeringAngle(steering_angle)
    
    # 車種によって Throttle のみで動かない場合に備え、目標速度(Cruise Speed)も併用
    target_speed_kmh = throttle * 40.0  # アクセル全開で 40 km/h
    driver.setCruisingSpeed(target_speed_kmh)
    driver.setBrakeIntensity(brake)

    # --- 自車状態の取得 ---
    gps_values = gps.getValues()  # [x, y, z]
    ego_pos_xz = (gps_values[0], gps_values[2])
    ego_speed_mps = update_ego_speed_mps(ego_pos_xz, sim_time_sec)
    distance_to_intersection_m = distance_2d(ego_pos_xz, Config.INTERSECTION_POSITION_XZ)

    # --- 歩行者の飛び出し再現(blind_spot_pedestrian_supervisor.py と同じロジック) ---
    distance_to_pedestrian_m = float("nan")
    if supervisor_ready:
        if not pedestrian_triggered and distance_to_intersection_m <= Config.PEDESTRIAN_TRIGGER_DISTANCE_M:
            pedestrian_triggered = True
            print(f"[INFO] 交差点まで{distance_to_intersection_m:.1f}m。歩行者が飛び出します。")

        if pedestrian_triggered and pedestrian_walked_distance_m < Config.PEDESTRIAN_WALK_DISTANCE_M:
            dt_sec = timestep / 1000.0
            step_dist_m = Config.PEDESTRIAN_WALK_SPEED_MPS * dt_sec
            current_pos = pedestrian_translation_field.getSFVec3f()
            new_pos = [
                current_pos[0] + Config.PEDESTRIAN_WALK_DIRECTION_XZ[0] * step_dist_m,
                current_pos[1],
                current_pos[2] + Config.PEDESTRIAN_WALK_DIRECTION_XZ[1] * step_dist_m,
            ]
            pedestrian_translation_field.setSFVec3f(new_pos)
            pedestrian_walked_distance_m += step_dist_m

        pedestrian_pos = pedestrian_translation_field.getSFVec3f()
        distance_to_pedestrian_m = distance_2d(ego_pos_xz, (pedestrian_pos[0], pedestrian_pos[2]))

    # --- 記録 ---
    if recording:
        csv_writer.writerow([
            f"{sim_time_sec:.3f}",
            f"{gps_values[0]:.3f}", f"{gps_values[1]:.3f}", f"{gps_values[2]:.3f}",
            f"{ego_speed_mps:.3f}",
            f"{distance_to_intersection_m:.3f}",
            f"{distance_to_pedestrian_m:.3f}" if supervisor_ready else "",
            f"{throttle:.3f}",
            f"{brake:.3f}",
            f"{steering_angle:.4f}",
        ])

    if frame_index % 30 == 0:
        rec_str = f"REC(take={take_count:04d})" if recording else "----"
        print(
            f"[{rec_str}] t={sim_time_sec:6.2f}s speed={ego_speed_mps:.2f}m/s "
            f"交差点まで={distance_to_intersection_m:.1f}m "
            f"歩行者まで={distance_to_pedestrian_m:.1f}m "
            f"throttle={throttle:.2f} brake={brake:.2f} steer={steering_angle:.3f}"
        )

    frame_index += 1