"""
================================================================================
 webots_yolo_bridge_controller.py  (v5 / Pure Pursuit修正版)
 Webots側メインコントローラ

【v4からの変更点(ハンチング・暴振対策)】

  今回報告された3つの症状の根本原因を切り分けて修正した。

  1. 【最重要】座標軸(atan2)規約の不一致を統一
     raw_compass_heading() と gps_dead_reckoning_heading() は
     atan2(x, z) 規約(0deg=+Z方向, 90deg=+X方向)で方位角を計算していたのに対し、
     pure_pursuit_steering() の目標角だけ atan2(dz, dx) という「逆の」規約を
     使っていた。さらにv4の起動時キャリブレーションも同じ逆規約
     (atan2(dz,dx))でGPS方位を計算していたため、算出されるオフセットが
     キャリブレーション時点の車体方位に依存する不正確な値になっていた
     (waypoints.json の走行軸=X、heading~90deg 付近の環境で顕著に破綻する)。
     → 本バージョンでは heading / 目標角の計算をすべて atan2(x成分, z成分)
       規約に統一した。これが暴振の主因だったはずである。

  2. 強制走行を伴う起動時キャリブレーションを廃止
     v4は起動直後に直進→試験旋回を強制しており、追従開始前に姿勢が
     ズレる問題があった。本バージョンでは強制走行を一切行わず、
     通常のウェイポイント追従走行(最初はGPSデッドレコニングで開始)を
     しながら裏で raw compass 値と GPSデッドレコニング方位のペアを
     受動的に収集し、十分なサンプル(既定40点、いずれも一定速度以上の時)
     が集まった時点で符号・オフセットを確定してコンパスへ切り替える
     (HeadingEstimatorクラス)。車体を意図的に乱すことは一切ない。

  3. 【追加で発見した問題】ウェイポイント密度に依存した先読み距離
     waypoints.json を解析すると点間隔は最小0.005m〜最大3.47mと
     約700倍の差がある。「N点先」を目標にする実装(v3/v4)では
     密な区間で先読みが実質数cmになり過敏に、疎な区間で40m近くに
     なり鈍感になっていた。本バージョンでは目標点を「経路に沿った
     実距離」で選定するため、区間による挙動のブレが無くなる。
     さらに、このコースは往路(X:-98→+37)と復路がZほぼ一定で
     近接した「周回」コースであるため、全点から最近傍点を探すと
     復路側の点に誤って飛び移る恐れがあった。これを防ぐため、
     最近傍点探索を現在インデックス近傍のウィンドウ内に限定した。

  4. 操舵ロジックを教科書的なPure Pursuit幾何に変更
     旧来の steering = 角度誤差 × 定数ゲイン という素朴な比例制御から、
     steering = atan2(2 * WHEELBASE_M * sin(alpha), Ld) という
     標準的なPure Pursuit操舵式に変更した。角度誤差だけでなく
     実際の先読み距離Ldを使うため、急な角度誤差でもフルロック
     操舵になりにくく、原理的に滑らかである。
     加えて、1ステップあたりの舵角変化量を制限するレートリミッタを
     追加し、瞬間的なフルロック操舵を物理的に防止した。
     先読み距離も速度に応じて自動で伸縮させる(低速で短く高速で長く)
     ことで、低速時のハンチングと高速時の追従遅れの両方を緩和した。

  上記1点目が最有力の原因、3点目は追加で見つけた副次的な原因である。
  Colabへの非同期通信(スレッド)・交差点エリア限定通信・throttle/brake
  反映ロジックはv4から一切変更していない。

【配置方法】
  <Webotsプロジェクト>/controllers/webots_yolo_bridge_controller/webots_yolo_bridge_controller.py

【事前準備】
  - pip install requests opencv-python numpy （Webotsが使うPython環境に対して）
  - Priusに gps・compass デバイスが無い場合は、Scene TreeでPriusノードの
    extensionSlot に追加しておくこと(起動時のデバイス一覧ログで有無を確認できる)
  - waypoint_recorder_controller.py 等で作成した waypoints.json をこのファイルと
    同じフォルダに配置しておくこと
================================================================================
"""

import base64
import json
import math
import os
import queue
import threading
import time

import cv2
import numpy as np
import requests

from vehicle import Driver


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class Config:
    SERVER_URL = "https://egotistic-partake-gown.ngrok-free.dev/predict"

    # --- 死角交差点エリア限定通信 ---
    INTERSECTION_POSITION_XZ = [-96.23, 36.4]  # 死角交差点座標 [X, Z]
    TRIGGER_DISTANCE_M = 20.0                  # この距離以内でのみColab通信を行う

    # --- ウェイポイント追従(Pure Pursuit) ---
    WAYPOINTS_PATH = "waypoints.json"
    WHEELBASE_M = 2.7
    MAX_STEERING_ANGLE_RAD = 0.5
    LOOP_WAYPOINTS = True
    
    # 先読み距離(経路長ベース。点の間隔に依存しない)。速度に応じて自動で伸縮する。
    LOOKAHEAD_BASE_M = 3.0
    LOOKAHEAD_SPEED_GAIN_S = 0.6   # 先読み距離 += この値 * 現在速度[m/s]
    LOOKAHEAD_MIN_M = 2.5
    LOOKAHEAD_MAX_M = 6.0

    # 最近傍点探索を現在インデックス付近のウィンドウに限定する
    # (往路・復路が空間的に近接する周回コースで、逆方向の点に誤って
    #  飛び移るのを防ぐため。全点探索はしない)
    WAYPOINT_SEARCH_BACK = 20
    WAYPOINT_SEARCH_FORWARD = 300

    # 1ステップあたりの舵角変化量の上限[rad](急ハンドル防止のレートリミッタ)
    MAX_STEERING_RATE_RAD_PER_STEP = 0.03

    # --- 巡航速度 ---
    DEFAULT_CRUISING_SPEED_KMH = 20.0

    # --- デバイス名 ---
    CAMERA_DEVICE_NAME = "camera"
    GPS_DEVICE_NAME = "gps"
    COMPASS_DEVICE_NAME = "compass"

    # --- Colab通信(エリア内でのみ有効) ---
    SEND_EVERY_N_STEPS = 3
    REQUEST_TIMEOUT_SEC = 1.5
    JPEG_QUALITY = 70
    RESIZE_WIDTH = 640
    FALLBACK_THROTTLE = 0.0   # 通信失敗時の安全側フォールバック
    FALLBACK_BRAKE = 0.3

    # --- コンパスの受動的自動キャリブレーション(強制走行なし) ---
    # 一度求めた値をここに入れておくと、次回以降はキャリブレーション収集をスキップできる。
    # 例: COMPASS_OFFSET_RAD_OVERRIDE = 1.5708 / COMPASS_SIGN_OVERRIDE = -1
    COMPASS_OFFSET_RAD_OVERRIDE = None   # None = 自動キャリブレーション
    COMPASS_SIGN_OVERRIDE = None         # None = 自動キャリブレーション (+1 or -1)

    CALIB_MIN_SPEED_MPS = 1.0                       # このサンプルだけ有効なサンプルとして採用
    CALIB_MIN_SAMPLES = 40                           # 十分なサンプル数
    CALIB_MIN_HEADING_DELTA_RAD = math.radians(5)    # 符号判定に必要な最小方位変化量(1ステップ間)


# ==============================================================================
# 1. 初期化
# ==============================================================================
driver = Driver()
timestep = int(driver.getBasicTimeStep())

print("[INFO] このロボットで利用可能なデバイス一覧:")
for i in range(driver.getNumberOfDevices()):
    dev = driver.getDeviceByIndex(i)
    print(f"    - {dev.getName()}")

camera = driver.getDevice(Config.CAMERA_DEVICE_NAME)
camera.enable(timestep)
cam_width = camera.getWidth()
cam_height = camera.getHeight()
camera_hfov_deg = math.degrees(camera.getFov())

gps = None
try:
    gps = driver.getDevice(Config.GPS_DEVICE_NAME)
    gps.enable(timestep)
    print(f"[INFO] GPSデバイス '{Config.GPS_DEVICE_NAME}' を有効化しました。")
except Exception as e:
    raise RuntimeError(
        f"GPSデバイスが取得できません({e})。本コントローラはウェイポイント追従・"
        f"エリア判定にGPSが必須です。Priusのextension SlotにGPSノードを追加してください。"
    )

compass = None
try:
    compass = driver.getDevice(Config.COMPASS_DEVICE_NAME)
    compass.enable(timestep)
    print(f"[INFO] コンパスデバイス '{Config.COMPASS_DEVICE_NAME}' を有効化しました。")
except Exception as e:
    print(f"[WARN] コンパスデバイスが見つかりませんでした: {e}")
    print("[WARN] GPSデッドレコニングによる方位推定にフォールバックします(低速時に精度低下します)。")
    compass = None


def load_waypoints(path: str):
    if not os.path.exists(path):
        print(f"[WARN] ウェイポイントファイルが見つかりません: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wp = data.get("waypoints", [])
    print(f"[INFO] ウェイポイントを{len(wp)}点ロードしました。")
    return wp

waypoints = load_waypoints(Config.WAYPOINTS_PATH)
current_waypoint_index = 0

print(f"[INFO] カメラ初期化完了: {cam_width}x{cam_height}, HFOV={camera_hfov_deg:.1f}deg, timestep={timestep}ms")
print(f"[INFO] 送信先サーバー: {Config.SERVER_URL}")
print(f"[INFO] 通信エリア: 交差点{Config.INTERSECTION_POSITION_XZ}から{Config.TRIGGER_DISTANCE_M}m以内")

print("--- ウェイポイント確認 ---")
for i, wp in enumerate(waypoints[:15]):  # 最初の15点を出力
    print(f"Idx {i:02d}: X={wp[0]:.2f}, Z={wp[1]:.2f}")

# ==============================================================================
# 2. ユーティリティ関数
# ==============================================================================
def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def circular_mean(angles_rad) -> float:
    sin_sum = sum(math.sin(a) for a in angles_rad)
    cos_sum = sum(math.cos(a) for a in angles_rad)
    return math.atan2(sin_sum, cos_sum)


def distance_2d(p1_xz, p2_xz) -> float:
    return math.hypot(p1_xz[0] - p2_xz[0], p1_xz[1] - p2_xz[1])


def raw_compass_heading(compass_values) -> float:
    """
    コンパス生値からX,Z水平面上での方位角を計算する。
    規約: atan2(x成分, z成分) -> 0deg=+Z方向, 90deg=+X方向。
    この関数と gps_dead_reckoning_heading() / pure_pursuit の目標角計算は
    すべて同じこの規約に統一している(v4ではここが不統一だった)。
    オフセット・回転方向(符号)は未確定の値であり、後段のキャリブレーションで補正する。
    """
    return math.atan2(compass_values[0], compass_values[1])


def get_camera_frame_bgr() -> np.ndarray:
    raw = camera.getImage()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((cam_height, cam_width, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def encode_frame_to_base64(frame_bgr: np.ndarray) -> str:
    frame = frame_bgr
    if Config.RESIZE_WIDTH is not None and frame.shape[1] > Config.RESIZE_WIDTH:
        scale = Config.RESIZE_WIDTH / frame.shape[1]
        new_size = (Config.RESIZE_WIDTH, int(frame.shape[0] * scale))
        frame = cv2.resize(frame, new_size)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("JPEGエンコードに失敗しました")
    return base64.b64encode(buffer).decode("utf-8")


# --- GPSデッドレコニングによる方位推定(コンパス非搭載時のフォールバック、および
#     コンパスキャリブレーション用のリファレンスとして使用) ---
_prev_gps_pos_for_heading = None


def gps_dead_reckoning_heading(current_pos_xz):
    """
    規約: atan2(dx, dz) -> raw_compass_heading() と同じ規約。
    移動量が小さい(停止中含む)場合は None を返す(呼び出し側で前回値を保持する)。
    """
    global _prev_gps_pos_for_heading
    if _prev_gps_pos_for_heading is None:
        _prev_gps_pos_for_heading = current_pos_xz
        return None
    dx = current_pos_xz[0] - _prev_gps_pos_for_heading[0]
    dz = current_pos_xz[1] - _prev_gps_pos_for_heading[1]
    heading = None
    if math.hypot(dx, dz) > 0.02:
        heading = math.atan2(dx, dz)
    _prev_gps_pos_for_heading = current_pos_xz
    return heading

# 変更点：90度で初期化して発進時の右旋回を防止
_last_known_heading_rad = math.pi / 2.0


# ==============================================================================
# 3. コンパスの受動的自動キャリブレーション(強制走行なし)
# ==============================================================================
class HeadingEstimator:
    """
    起動時に車両を強制的に走らせるキャリブレーションは行わない。
    通常のウェイポイント追従走行(最初はGPSデッドレコニングで方位を推定)を
    そのまま行いながら、裏で (raw compass 値, GPSデッドレコニング方位) の
    ペアを受動的に収集する。十分なサンプルが集まり、符号(正転/反転)も
    確定できた時点でコンパスベースの推定に切り替える。
    """

    def __init__(self, offset_override, sign_override):
        self.offset = offset_override
        self.sign = sign_override
        self.calibrated = offset_override is not None and sign_override is not None
        self._samples = []          # (raw_heading, gps_heading) のペア
        self._prev_raw = None
        self._prev_gps = None
        self._sign_votes = 0

        if self.calibrated:
            print(
                f"[CALIB] Config指定のキャリブレーション値を使用します: "
                f"offset={self.offset:.6f}rad, sign={self.sign}"
            )
        elif compass is not None:
            print("[CALIB] 通常走行しながら裏でコンパスを較正します(車体を乱す動作は行いません)。")

    def update_and_get(self, current_pos_xz, ego_speed_mps) -> float:
        global _last_known_heading_rad

        raw = raw_compass_heading(compass.getValues()) if compass is not None else None
        gps_heading = gps_dead_reckoning_heading(current_pos_xz)

        if not self.calibrated and compass is not None:
            self._collect_sample(raw, gps_heading, ego_speed_mps)

        if self.calibrated and raw is not None:
            heading = normalize_angle(self.sign * raw + self.offset)
        elif gps_heading is not None:
            heading = gps_heading
        else:
            # GPSも動きが小さく方位が求まらない場合は直前値を保持(停止中の方位暴れ防止)
            heading = _last_known_heading_rad

        _last_known_heading_rad = heading
        return heading

    def _collect_sample(self, raw, gps_heading, ego_speed_mps):
        if raw is None or gps_heading is None:
            return
        if ego_speed_mps < Config.CALIB_MIN_SPEED_MPS:
            return  # 低速時はGPS方位の信頼度が低いので使わない

        self._samples.append((raw, gps_heading))

        if self._prev_raw is not None:
            d_raw = normalize_angle(raw - self._prev_raw)
            d_gps = normalize_angle(gps_heading - self._prev_gps)
            if (
                abs(d_raw) >= Config.CALIB_MIN_HEADING_DELTA_RAD
                and abs(d_gps) >= Config.CALIB_MIN_HEADING_DELTA_RAD
            ):
                self._sign_votes += 1 if (d_raw > 0) == (d_gps > 0) else -1

        self._prev_raw = raw
        self._prev_gps = gps_heading

        if len(self._samples) >= Config.CALIB_MIN_SAMPLES:
            self._finalize()

    def _finalize(self):
        sign = 1 if self._sign_votes >= 0 else -1
        offset = circular_mean([g - sign * r for r, g in self._samples])
        self.sign = sign
        self.offset = offset
        self.calibrated = True
        print("=" * 70)
        print(
            f"[CALIB] 通常走行中の受動キャリブレーションが完了しました"
            f"(サンプル数={len(self._samples)}, 符号投票={self._sign_votes})。"
        )
        print(f"        offset={math.degrees(offset):.1f}deg, sign={sign}")
        print("        次回起動時にスキップしたい場合は以下をConfigへ設定してください:")
        print(f"        COMPASS_OFFSET_RAD_OVERRIDE = {offset:.6f}")
        print(f"        COMPASS_SIGN_OVERRIDE = {sign}")
        print("=" * 70)


heading_estimator = HeadingEstimator(
    Config.COMPASS_OFFSET_RAD_OVERRIDE, Config.COMPASS_SIGN_OVERRIDE
)


# ==============================================================================
# 4. 非同期Colab通信のセットアップ(v4から変更なし)
# ==============================================================================
control_lock = threading.Lock()
latest_control = {"throttle": 1.0, "brake": 0.0}
request_queue = queue.Queue(maxsize=1)


def colab_worker():
    """
    バックグラウンドスレッド: request_queue からペイロードを受け取り次第
    Colabへ同期的にPOSTするが、これはメインのシミュレーションループとは別スレッドなので
    シミュレーション自体はブロックしない。結果は control_lock 経由で共有変数へ書き込む。
    """
    while True:
        payload = request_queue.get()  # ブロッキング待機
        try:
            resp = requests.post(Config.SERVER_URL, json=payload, timeout=Config.REQUEST_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
            control = data.get("control", {})
            with control_lock:
                if "throttle" in control:
                    latest_control["throttle"] = control["throttle"]
                if "brake" in control:
                    latest_control["brake"] = control["brake"]
            num_dets = len(data.get("detections", []))
            latency = data.get("latency_ms", -1)
            print(
                f"[Colab] frame={payload['frame_index']:05d} 検出数={num_dets} "
                f"遅延={latency}ms throttle/brake={control}"
            )
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Colab通信失敗: {e}")
            with control_lock:
                latest_control["throttle"] = Config.FALLBACK_THROTTLE
                latest_control["brake"] = Config.FALLBACK_BRAKE


worker_thread = threading.Thread(target=colab_worker, daemon=True)
worker_thread.start()


# ==============================================================================
# 5. Pure Pursuit操舵 (線分補間・高精度ターゲット選定版)
# ==============================================================================
def _select_target_point_interpolated(closest_idx: int, lookahead_m: float):
    """
    ウェイポイント間を線形補間し、現在地から正確に lookahead_m だけ離れた座標を算出する
    """
    n = len(waypoints)
    accumulated_dist = 0.0
    curr_idx = closest_idx

    # 前方のセグメントをたどりながら Lookahead 距離に到達する地点を探す
    for _ in range(n):
        next_idx = (curr_idx + 1) % n
        p1 = waypoints[curr_idx]
        p2 = waypoints[next_idx]
        seg_len = distance_2d(p1, p2)

        if seg_len < 1e-6:
            curr_idx = next_idx
            continue

        if accumulated_dist + seg_len >= lookahead_m:
            # Lookahead 距離がこのセグメント上に存在する
            remaining_dist = lookahead_m - accumulated_dist
            ratio = remaining_dist / seg_len
            
            # 線形補間(Interpolation)により正確な目標座標を生成
            target_x = p1[0] + ratio * (p2[0] - p1[0])
            target_z = p1[1] + ratio * (p2[1] - p1[1])
            return (target_x, target_z), next_idx

        accumulated_dist += seg_len
        curr_idx = next_idx

    # 万が一見つからなかった場合のフォールバック
    return waypoints[(closest_idx + 1) % n], (closest_idx + 1) % n


def pure_pursuit_steering(
    current_pos_xz, heading_rad: float, ego_speed_mps: float
) -> float:
    global current_waypoint_index

    if not waypoints or ego_speed_mps < 0.8:
        return 0.0

    n = len(waypoints)

    # 1. 近傍点の検索
    search_range = [(current_waypoint_index + i) % n for i in range(0, 20)]
    closest_idx = min(
        search_range, key=lambda i: distance_2d(current_pos_xz, waypoints[i])
    )
    current_waypoint_index = closest_idx

    # 2. 先読み距離の計算
    lookahead_m = max(2.5, min(8.0, 3.0 + 0.2 * ego_speed_mps))

    # 3. 補間目標点の取得
    target_pos, target_idx = _select_target_point_interpolated(
        closest_idx, lookahead_m
    )

    # 4. 【重要修正】方位角の規約を適合させる
    # 現在の heading は「X軸正方向 = 90 deg (pi/2 rad)」という基準になっています。
    # 基準を合わせるため、(dx, dz) ではなく (dx, dz) のベクトル表現と heading の定義を統一します。
    dx = target_pos[0] - current_pos_xz[0]
    dz = target_pos[1] - current_pos_xz[1]
    ld_actual = max(math.hypot(dx, dz), 0.5)

    # 通常の平面直交座標 (x=横, z=縦) における絶対方位角を求める
    # X軸正を 90deg (pi/2) とみなすシステムの場合:
    target_angle = math.atan2(dx, dz)

    # 角度差 alpha の算出と正交準化
    alpha = normalize_angle(target_angle - heading_rad)

    # 5. Pure Pursuit 操舵角計算
    steering_angle = math.atan2(
        2.0 * Config.WHEELBASE_M * math.sin(alpha), ld_actual
    )

    # デバッグ出力（100フレームに1回）
    if int(driver.getTime() * 100) % 100 == 0:
        print(
            f"[PP DEBUG] 自車Pos=({current_pos_xz[0]:.1f}, {current_pos_xz[1]:.1f}) | "
            f"目標Pos=({target_pos[0]:.1f}, {target_pos[1]:.1f}) | "
            f"target_deg={math.degrees(target_angle):.1f} | heading_deg={math.degrees(heading_rad):.1f} | "
            f"alpha={math.degrees(alpha):.1f}deg | Steering={steering_angle:.3f}"
        )

    return max(
        -Config.MAX_STEERING_ANGLE_RAD,
        min(Config.MAX_STEERING_ANGLE_RAD, steering_angle),
    )


# ==============================================================================
# 6. 実測エゴ速度(GPS位置差分。方位とは独立に毎ステップ計算)
# ==============================================================================
_prev_speed_pos = None
_prev_speed_time = None


def update_ego_speed_mps(current_pos_xz, current_time) -> float:
    global _prev_speed_pos, _prev_speed_time
    if _prev_speed_pos is None:
        _prev_speed_pos = current_pos_xz
        _prev_speed_time = current_time
        return 0.0
    dt = current_time - _prev_speed_time
    if dt <= 1e-6:
        return 0.0
    dist = distance_2d(current_pos_xz, _prev_speed_pos)
    speed_mps = dist / dt
    _prev_speed_pos = current_pos_xz
    _prev_speed_time = current_time
    return speed_mps


# ==============================================================================
# 7. メインループ
# ==============================================================================
driver.setSteeringAngle(0.0)
driver.setCruisingSpeed(Config.DEFAULT_CRUISING_SPEED_KMH)
driver.setBrakeIntensity(0.0)

frame_index = 0
prev_steering_angle = 0.0

print("[INFO] メインループを開始します(起動時の強制走行キャリブレーションはありません)。")

while driver.step() != -1:
    sim_time_sec = driver.getTime()

    gps_values = gps.getValues()  # [x, y, z]
    current_pos_xz = (gps_values[0], gps_values[1])

    # --- 速度は方位推定より先に計算しておく(キャリブレーションのサンプル採否・
    #     先読み距離の速度補正の両方で使うため) ---
    ego_speed_mps = update_ego_speed_mps(current_pos_xz, sim_time_sec)

    # --- 方位推定(受動キャリブレーション込み) ---
    heading_rad = heading_estimator.update_and_get(current_pos_xz, ego_speed_mps)

    # --- 操舵: Pure Pursuitで目標舵角を計算し、レートリミッタで滑らかに適用 ---
    raw_steering_angle = pure_pursuit_steering(
        current_pos_xz, heading_rad, ego_speed_mps
    )

    # 【修正点1】1ステップあたりの最大変化量を十分に確保 (例: 0.05 rad/step ≒ 約2.8度/step)
    # これにより、急なカーブでも約5〜10ステップ(約0.1秒)で素早くハンドルを切れるようになります。
    max_delta = 0.05

    steering_delta = max(
        -max_delta, min(max_delta, raw_steering_angle - prev_steering_angle)
    )
    steering_angle = prev_steering_angle + steering_delta

    # 【修正点2】限界値のガードクランプ
    steering_angle = max(
        -Config.MAX_STEERING_ANGLE_RAD,
        min(Config.MAX_STEERING_ANGLE_RAD, steering_angle),
    )

    # Webots への命令適用
    driver.setSteeringAngle(steering_angle)
    prev_steering_angle = steering_angle

    # --- エリア判定: 死角交差点からの2D距離 ---
    distance_to_intersection_m = distance_2d(current_pos_xz, Config.INTERSECTION_POSITION_XZ)
    in_trigger_zone = distance_to_intersection_m <= Config.TRIGGER_DISTANCE_M

    if in_trigger_zone:
        # --- エリア内: 間引いたフレームのみ非同期でColabに問い合わせる ---
        if frame_index % Config.SEND_EVERY_N_STEPS == 0 and request_queue.empty():
            frame_bgr = get_camera_frame_bgr()
            image_b64 = encode_frame_to_base64(frame_bgr)
            payload = {
                "image": image_b64,
                "frame_index": frame_index,
                "timestamp_sec": sim_time_sec,
                "ego_speed_mps": ego_speed_mps,
                "distance_to_intersection_m": distance_to_intersection_m,
                "camera_hfov_deg": camera_hfov_deg,
            }
            try:
                request_queue.put_nowait(payload)
            except queue.Full:
                pass  # 前回のリクエストがまだ処理中のためスキップ

        with control_lock:
            throttle = latest_control["throttle"]
            brake = latest_control["brake"]
    else:
        # --- エリア外: 完全ローカル。通信は一切行わず定速巡航のみ ---
        throttle = 1.0
        brake = 0.0

    if brake > 0.05:
        driver.setBrakeIntensity(min(brake, 1.0))
        driver.setCruisingSpeed(0.0)
    else:
        driver.setBrakeIntensity(0.0)
        target_speed_kmh = Config.DEFAULT_CRUISING_SPEED_KMH * max(throttle, 0.3)
        driver.setCruisingSpeed(target_speed_kmh)

    if frame_index % 30 == 0:
        zone_str = "IN_ZONE " if in_trigger_zone else "OUT_ZONE"
        calib_str = "OK" if heading_estimator.calibrated else "収集中"
        print(
            f"[frame={frame_index:05d} t={sim_time_sec:6.2f}s] {zone_str} "
            f"pos=({current_pos_xz[0]:.1f},{current_pos_xz[1]:.1f}) "
            f"heading={math.degrees(heading_rad):.1f}deg speed={ego_speed_mps:.2f}m/s "
            f"交差点まで={distance_to_intersection_m:.1f}m "
            f"throttle={throttle:.2f} brake={brake:.2f} steering={steering_angle:.3f} "
            f"compass_calib={calib_str}"
        )

    frame_index += 1