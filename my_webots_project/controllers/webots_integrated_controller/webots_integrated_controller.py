import math
import os
import warnings
import threading
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # ★ 背景画像読み込み用に追加
from matplotlib.widgets import Button

from vehicle import Driver

# 経路探索モジュール ＆ OpenDRIVEパーサ
from route_planner import (
    find_route,
    route_to_raw_waypoints,
    extract_stop_trigger_positions,
    RouteTriggerManager,
)
from opendrive_parser import parse_opendrive, build_road_graph, generate_sample_xodr

# --- OpenDRIVEパーサを使ってグラフを動的生成 ---
_odr_text = generate_sample_xodr()
_odr_map = parse_opendrive(_odr_text, from_string=True)
ROAD_GRAPH = build_road_graph(_odr_map, start_road_id="1", goal_road_id="7")

# ★ グラフから自動でスタート候補（最初に見つかったノード）とゴール候補を取得する
_node_ids = list(ROAD_GRAPH["nodes"].keys())
AUTO_DEFAULT_START = _node_ids[0] if _node_ids else "J0"
AUTO_DEFAULT_GOAL = _node_ids[-1] if len(_node_ids) > 1 else "J7"

try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_LIB_AVAILABLE = True
except ImportError:
    YOLO_LIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================
# 定数定義
# ============================================================
WHEELBASE = 2.7          # Pure Pursuit用ホイールベース[m]
LOOKAHEAD_MIN = 4.5      # 最小Lookahead距離[m]
LOOKAHEAD_GAIN = 0.3     # 速度比例ゲイン
STEERING_LIMIT = 0.5     # 操舵角制限[rad]
LOG_EVERY_N_STEPS = 10
V_TABLE_PATH = "v_table_for_webots.npz"

# ---- カメラ & YOLO検出 関連定数 ----
CAMERA_DEVICE_NAME = "camera"
YOLO_MODEL_PATH = "yolov8n.pt"           # 標準モデル
YOLO_ONNX_PATH = "yolov8n.onnx"          # 高速化用ONNXモデル
YOLO_CONF_THRESHOLD = 0.4                # 検知信頼度の閾値

CLASS_ID_PERSON = 0          # 人間
CLASS_ID_TRAFFIC_LIGHT = 9   # 信号機
CLASS_ID_STOP_SIGN = 11      # 停止標識

PEDESTRIAN_REAL_HEIGHT_M = 1.7           # 歩行者の想定身長[m]

# ---- RSS 関連定数 ----
RSS_REACTION_TIME = 1.0     # 応答時間 ρ [s]
RSS_MAX_ACCEL = 1.0         # 最大加速度 [m/s^2]
RSS_MIN_BRAKE = 4.0         # 最小制動減速度 [m/s^2]

STATE_WAITING_FALLBACK_TIMEOUT_SEC = 8.0
MIN_MANDATORY_WAIT_SEC = 4.5  # 4.5秒の最低待機時間

DEFAULT_START_NODE_ID = AUTO_DEFAULT_START  
DEFAULT_GOAL_NODE_ID = AUTO_DEFAULT_GOAL    

STOP_TRIGGER_MARGIN_M = 2.5   # 2.5 -> 4.0と、値を大きくして交差点手前できちんと減速+停止を実行  
RE_ACCEL_RESUME_CRUISE_SPEED_KMH = 15.0
CRUISE_SPEED_KMH = 40.0

# --- 【修正】ゴール接近時の滑らかな減速用パラメータ ---
GOAL_DECEL_START_DIST_M = 15.0   # この距離を切ったら減速を開始する
GOAL_STOP_DIST_M = 2.0           # この距離で完全停止とみなし、STATE_ARRIVEDへ固定する

# ============================================================
# ステート定数の定義（FSM）
# ============================================================
STATE_CRUISE = 1
STATE_STOP_INTERSECTION = 2
STATE_WAITING = 3
STATE_RE_ACCELERATE = 4
STATE_ARRIVED = 5   # 【追加】ゴール到着後、速度0を維持し続ける状態（simは終了させない）


# ============================================================
# インタラクティブ・ゴール選択 GUI (matplotlib) [背景画像対応版]
# ============================================================
class InteractiveRouteSelector:
    def __init__(self, road_graph, default_start="N1", default_goal="N8"):
        self.road_graph = road_graph
        self.start_node = default_start
        self.selected_goal = default_goal
        self.confirmed = False

        # ノード情報の取得元を自動判定
        self.nodes_pos = {}
        if isinstance(road_graph, dict):
            if "nodes" in road_graph:
                self.nodes_pos = road_graph["nodes"]
            else:
                self.nodes_pos = {k: v for k, v in road_graph.items() if k != "edges"}
        elif hasattr(road_graph, "nodes"):
            self.nodes_pos = road_graph.nodes

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.ax.set_title("Webots Autonomous Route Selector\n[Click node to set GOAL, then click Start button]", fontsize=12)
        self.ax.set_xlabel("X Position [m]")
        self.ax.set_ylabel("Y Position [m]")
        self.ax.grid(True, linestyle="--", alpha=0.6)

        # ==========================================================
        # ★【追加】Webotsの鳥瞰図画像を背景として読み込み・表示する
        # ==========================================================
        try:
            # 同じフォルダに置いた画像ファイル名に合わせて変更してください
            bg_img = mpimg.imread("world_top_view.png")
            
            # 【重要】OpenDRIVEの道路座標と画像のピクセル範囲 (X, Yの範囲)
            # マップに合わせて必要に応じて数値を調整してください
            x_min, x_max = -100.0, 100.0
            y_min, y_max = -100.0, 100.0
            
            self.ax.imshow(bg_img, extent=[x_min, x_max, y_min, y_max], origin='upper', alpha=0.7)
            print("[GUI] 背景画像 'world_top_view.png' の読み込みに成功しました。")
        except FileNotFoundError:
            print("[GUI][WARNING] 背景画像が見つからないため、通常モードで起動します。")
        # ==========================================================

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 確定ボタンの設置
        ax_button = plt.axes([0.7, 0.02, 0.25, 0.07])
        self.btn_start = Button(ax_button, '🚀 走行開始', color='lightgoldenrodyellow', hovercolor='0.9')
        self.btn_start.on_clicked(self.on_submit)

        self.update_plot()

    def _get_node_xy(self, node_id):
        """ノードの構造を自動解析し、尺度（スケール）とオフセットを調整して返す"""
        if node_id not in self.nodes_pos:
            return None
        val = self.nodes_pos[node_id]
        x, y = 0.0, 0.0
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
            x, y = float(val[0]), float(val[1])
        elif isinstance(val, dict):
            if "pos" in val:
                p = val["pos"]
                x, y = float(p[0]), float(p[1])
            elif "x" in val and "y" in val:
                x, y = float(val["x"]), float(val["y"])
        else:
            return None

        # ==========================================================
        # ★【スケール（拡大・縮小）とオフセットの調整】
        # - scale: 1.0より小さく（例: 0.85 など）すると全体が縮小します。
        #   逆に大きく（1.1など）すると拡大します。
        # - offset: 全体の平行移動量を微調整します。
        # ==========================================================
        scale = 0.60    # 尺度を少し小さくする（例: 0.8〜0.9くらいで調整）
        
        # スケールを適用する中心位置（必要に応じて変更。大体 0, 0 や画面中央付近）
        center_x = 0.0
        center_y = 0.0
        
        # 中心を基準にスケーリング
        x_scaled = center_x + (x - center_x) * scale
        y_scaled = center_y + (y - center_y) * scale

        # 平行移動（これまでの調整値を維持・微調整）
        offset_x = -2.0  
        offset_y = -3.0  
        
        return x_scaled + offset_x, y_scaled + offset_y

    def _parse_edge(self, edge):
        """エッジのデータ構造（辞書・タプル等）を自動判定して始点と終点を返す"""
        u, v = None, None
        if isinstance(edge, dict):
            for u_key in ['u', 'from', 'start', 0, 'node1']:
                if u_key in edge:
                    u = edge[u_key]
                    break
            for v_key in ['v', 'to', 'end', 1, 'node2']:
                if v_key in edge:
                    v = edge[v_key]
                    break
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            u, v = edge[0], edge[1]
        return u, v

    def update_plot(self):
        self.ax.clear()
        
        # 背景画像がある場合、クリアされてしまうため再描画時に再度imshowを呼び出すか、
        # あるいはタイトル等の再設定を行う
        try:
            bg_img = mpimg.imread("world_top_view.png")
            x_min, x_max = -100.0, 100.0
            y_min, y_max = -100.0, 100.0
            self.ax.imshow(bg_img, extent=[x_min, x_max, y_min, y_max], origin='upper', alpha=0.7)
        except FileNotFoundError:
            pass

        self.ax.set_title(f"Start: {self.start_node} | Selected Goal: {self.selected_goal}\n[Click node to change goal]", fontsize=11)
        self.ax.grid(True, linestyle="--", alpha=0.6)

        # 全エッジ（灰色）の描画
        edges = []
        if isinstance(self.road_graph, dict) and "edges" in self.road_graph:
            edges = self.road_graph["edges"]
        elif hasattr(self.road_graph, "edges"):
            edges = self.road_graph.edges

        for edge in edges:
            u, v = self._parse_edge(edge)
            p1 = self._get_node_xy(u)
            p2 = self._get_node_xy(v)
            if p1 and p2:
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=2, alpha=0.6)

        # 選択ルートの計算と黄色ハイライト
        try:
            route_res = find_route(self.road_graph, self.start_node, self.selected_goal)
            if route_res:
                node_path, edge_path = route_res
                for edge in edge_path:
                    u, v = self._parse_edge(edge)
                    p1 = self._get_node_xy(u)
                    p2 = self._get_node_xy(v)
                    if p1 and p2:
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gold', linewidth=5, zorder=3)
        except Exception:
            pass

        # ノードの描画
        for n_id in self.nodes_pos.keys():
            pos = self._get_node_xy(n_id)
            if pos:
                color = 'blue'
                if n_id == self.start_node:
                    color = 'green'
                elif n_id == self.selected_goal:
                    color = 'red'
                
                self.ax.scatter(pos[0], pos[1], s=120, color=color, zorder=4)
                self.ax.text(pos[0] + 0.5, pos[1] + 0.5, n_id, fontsize=10, fontweight='bold', color='black')

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        min_dist = float('inf')
        closest_node = None
        for n_id in self.nodes_pos.keys():
            pos = self._get_node_xy(n_id)
            if pos:
                dist = math.hypot(pos[0] - event.xdata, pos[1] - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = n_id

        if closest_node and min_dist < 15.0:
            if closest_node != self.start_node:
                self.selected_goal = closest_node
                print(f"[GUI] ゴールノードが変更されました: {self.selected_goal}")
                self.update_plot()

    def on_submit(self, event):
        print(f"[GUI] ルート確定！ スタート: {self.start_node} -> ゴール: {self.selected_goal}")
        self.confirmed = True
        plt.close(self.fig)

    def show(self):
        plt.show()
        return self.selected_goal


def densify_path(waypoints, spacing=0.5, lateral_offset=0.6):
    if not SCIPY_AVAILABLE or len(waypoints) < 4:
        return waypoints
    try:
        _, unique_indices = np.unique(waypoints, axis=0, return_index=True)
        unique_wps = waypoints[np.sort(unique_indices)]

        tck, _ = splprep([unique_wps[:, 0], unique_wps[:, 1]], s=0.5, k=3)
        dists = np.hypot(np.diff(unique_wps[:, 0]), np.diff(unique_wps[:, 1]))
        total_len = np.sum(dists)
        num_points = max(int(total_len / spacing), 2)

        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, tck)

        if lateral_offset != 0.0:
            dx = np.gradient(x_fine)
            dy = np.gradient(y_fine)
            norms = np.hypot(dx, dy)
            norms[norms == 0] = 1.0
            nx = -dy / norms
            ny = dx / norms
            x_fine += nx * lateral_offset
            y_fine += ny * lateral_offset

        return np.column_stack([x_fine, y_fine])
    except Exception:
        return waypoints


class LateralController:
    def __init__(self, waypoints, wheelbase=WHEELBASE):
        self.waypoints = waypoints
        self.wheelbase = wheelbase
        self.current_wp_idx = 0

        dists = np.hypot(np.diff(self.waypoints[:, 0]), np.diff(self.waypoints[:, 1]))
        self.remaining_distances = np.zeros(len(self.waypoints))
        self.remaining_distances[:-1] = np.cumsum(dists[::-1])[::-1]

    def update_current_index(self, curr_x, curr_y):
        dists = np.hypot(self.waypoints[:, 0] - curr_x, self.waypoints[:, 1] - curr_y)
        self.current_wp_idx = int(np.argmin(dists))

    def get_remaining_distance(self):
        return float(self.remaining_distances[self.current_wp_idx])

    def compute_steering(self, curr_x, curr_y, curr_yaw, curr_speed_ms):
        self.update_current_index(curr_x, curr_y)
        rem_dist = self.get_remaining_distance()

        if rem_dist < 2.0:
            return 0.0

        calc_lookahead = LOOKAHEAD_MIN + LOOKAHEAD_GAIN * max(curr_speed_ms, 0.0)
        lookahead = min(calc_lookahead, max(rem_dist * 0.8, 2.0))

        target_wp = self.waypoints[-1]
        for i in range(self.current_wp_idx, len(self.waypoints)):
            d = math.hypot(self.waypoints[i, 0] - curr_x, self.waypoints[i, 1] - curr_y)
            if d >= lookahead:
                target_wp = self.waypoints[i]
                break

        dx = target_wp[0] - curr_x
        dy = target_wp[1] - curr_y

        target_angle = math.atan2(dy, dx)
        alpha = target_angle - curr_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        gps_steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead)
        return float(np.clip(-gps_steering, -STEERING_LIMIT, STEERING_LIMIT))


class IntegratedAutonomousController:
    def __init__(self, start_node_id=DEFAULT_START_NODE_ID, goal_node_id=DEFAULT_GOAL_NODE_ID):
        self.driver = Driver()
        self.TIME_STEP = int(self.driver.getBasicTimeStep())

        self.gps = self.driver.getDevice("gps")
        if self.gps:
            self.gps.enable(self.TIME_STEP)

        self.imu = self.driver.getDevice("inertial unit")
        if self.imu:
            self.imu.enable(self.TIME_STEP)

        self.camera = self.driver.getDevice(CAMERA_DEVICE_NAME)
        if self.camera:
            self.camera.enable(self.TIME_STEP)
            self.camera.setFov(1.6)

        # YOLO初期化
        self.yolo_available = False
        self.yolo_model = None
        if YOLO_LIB_AVAILABLE and self.camera is not None:
            try:
                if not os.path.exists(YOLO_ONNX_PATH):
                    temp_model = YOLO(YOLO_MODEL_PATH)
                    temp_model.export(format="onnx", imgsz=320)
                self.yolo_model = YOLO(YOLO_ONNX_PATH)
                self.yolo_available = True
                print(f"[YOLO] ONNXモデルの読み込み成功")
            except Exception as e:
                print(f"[YOLO] 読み込み失敗: {e}")

        self.target_image_for_worker = None
        self.annotated_frame = None
        self.latest_detections = []
        self.inference_running = True
        self.lock = threading.Lock()

        if self.yolo_available:
            self.worker_thread = threading.Thread(target=self._yolo_worker, daemon=True)
            self.worker_thread.start()

        self._curr_yaw = 0.0
        self._yaw_offset = 0.0

        # ルート探索とウェイポイント生成
        route_result = find_route(ROAD_GRAPH, start_node_id, goal_node_id)
        if route_result is None:
            raise RuntimeError(
                f"[route_planner] ルートが見つかりません: {start_node_id} -> {goal_node_id}"
            )
        node_path, edge_path = route_result
        print(f"[route_planner] 探索ルート: {' -> '.join(node_path)}")

        raw_waypoints = route_to_raw_waypoints(ROAD_GRAPH, node_path, edge_path)
        dense_waypoints = densify_path(raw_waypoints, spacing=0.5)
        self.lateral = LateralController(dense_waypoints)

        stop_trigger_positions = extract_stop_trigger_positions(ROAD_GRAPH, node_path)
        self.trigger_manager = RouteTriggerManager(
            dense_waypoints,
            self.lateral.remaining_distances,
            stop_trigger_positions,
            trigger_margin=STOP_TRIGGER_MARGIN_M,
        )
        print(f"[route_planner] 停止トリガー数: {len(stop_trigger_positions)} 箇所")

        self._initialize_yaw(dense_waypoints)
        self.driver.setGear(1)

        self.current_state = STATE_CRUISE
        self.wait_timer = 0.0
        self.safe_consecutive_count = 0
        self.REQUIRED_SAFE_COUNT = 4
        self._arrived_logged = False   # 【追加】到着メッセージを1回だけ出すためのフラグ

    def _yolo_worker(self):
        while self.inference_running:
            with self.lock:
                img_to_process = self.target_image_for_worker
                self.target_image_for_worker = None

            if img_to_process is None:
                time.sleep(0.04)
                continue

            annotated_image = img_to_process.copy()
            detections = []

            if self.yolo_model is not None:
                try:
                    results = self.yolo_model.predict(
                        source=img_to_process,
                        classes=[CLASS_ID_PERSON, CLASS_ID_TRAFFIC_LIGHT, CLASS_ID_STOP_SIGN],
                        conf=YOLO_CONF_THRESHOLD,
                        imgsz=320,
                        verbose=False
                    )
                    for r in results:
                        annotated_image = r.plot()
                        if r.boxes is not None:
                            for box in r.boxes:
                                xyxy = box.xyxy[0].cpu().numpy()
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                detections.append({
                                    'class_id': cls_id,
                                    'bbox': (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                                    'confidence': conf
                                })
                except Exception:
                    pass

            with self.lock:
                self.annotated_frame = annotated_image
                self.latest_detections = detections

            time.sleep(0.12)

    def _estimate_pedestrian_distance(self, bbox):
        if self.camera is None:
            return None
        x1, y1, x2, y2 = bbox
        bbox_height_px = max(y2 - y1, 1e-3)
        try:
            fov = self.camera.getFov()
            img_height = self.camera.getHeight()
            focal_length_px = (img_height / 2.0) / math.tan(fov / 2.0)
            return float((PEDESTRIAN_REAL_HEIGHT_M * focal_length_px) / bbox_height_px)
        except Exception:
            return None

    def _initialize_yaw(self, dense_waypoints):
        self.driver.step()
        path_heading = math.atan2(
            dense_waypoints[1, 1] - dense_waypoints[0, 1],
            dense_waypoints[1, 0] - dense_waypoints[0, 0]
        )
        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            if rpy is not None and len(rpy) >= 3:
                raw_imu_yaw = rpy[2]
                self._yaw_offset = math.atan2(
                    math.sin(path_heading - raw_imu_yaw),
                    math.cos(path_heading - raw_imu_yaw)
                )
                self._curr_yaw = path_heading
                return
        self._yaw_offset = 0.0
        self._curr_yaw = path_heading

    def _get_vehicle_state(self):
        curr_x, curr_y = self.lateral.waypoints[0, 0], self.lateral.waypoints[0, 1]
        if self.gps is not None:
            gps_values = self.gps.getValues()
            if gps_values is not None and len(gps_values) >= 2:
                curr_x, curr_y = gps_values[0], gps_values[1]

        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            if rpy is not None and len(rpy) >= 3:
                self._curr_yaw = rpy[2] + self._yaw_offset

        dist = self.lateral.get_remaining_distance()
        curr_speed_kmh = self.driver.getCurrentSpeed() or 0.0

        return curr_x, curr_y, self._curr_yaw, dist, curr_speed_kmh / 3.6

    def _apply_actuation(self, target_speed_kmh, steering_angle):
        """
        【修正】以前は target_speed_kmh<=0.1 になった瞬間にbrakeIntensity=1.0
        （フルブレーキ）へ飛んでいたため、たとえrun()側で減速目標を作っても
        最後の一押しが常に急ブレーキになっていた。
        ここでは「目標速度との差（超過速度）」に応じて滑らかにブレーキを
        強めるようにし、完全停止付近（ほぼ0km/h かつ 目標0）でのみ
        フルブレーキにしてその場にしっかり停止を維持する。
        """
        self.driver.setGear(1)
        curr_speed_kmh = self.driver.getCurrentSpeed() or 0.0

        if target_speed_kmh > 0.1 and curr_speed_kmh < target_speed_kmh:
            # 加速
            self.driver.setBrakeIntensity(0.0)
            self.driver.setThrottle(0.6 if curr_speed_kmh < 1.0 else 0.5)
        else:
            # 減速 / 停止
            self.driver.setThrottle(0.0)
            overspeed_kmh = max(curr_speed_kmh - target_speed_kmh, 0.0)

            if target_speed_kmh <= 0.1 and curr_speed_kmh < 1.0:
                # 目標が完全停止で、かつほぼ止まっている -> しっかり踏んで保持する
                brake = 1.0
            else:
                # 超過速度に比例して滑らかにブレーキを強める（急停止を避ける）
                brake = float(np.clip(0.15 + 0.02 * overspeed_kmh, 0.15, 0.9))
            self.driver.setBrakeIntensity(brake)

        if math.isnan(steering_angle):
            steering_angle = 0.0

        self.driver.setSteeringAngle(steering_angle)

    def run(self):
        step_count = 0
        while self.driver.step() != -1:
            step_count += 1
            curr_x, curr_y, curr_yaw, curr_dist, curr_speed_ms = self._get_vehicle_state()

            steering_angle = self.lateral.compute_steering(
                curr_x, curr_y, curr_yaw, curr_speed_ms
            )

            if self.camera is not None and (step_count % 5 == 0):
                raw = self.camera.getImage()
                if raw is not None:
                    width = self.camera.getWidth()
                    height = self.camera.getHeight()
                    try:
                        img_bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
                        image_bgr = img_bgra[:, :, :3]
                        with self.lock:
                            if self.target_image_for_worker is None:
                                self.target_image_for_worker = image_bgr
                    except Exception:
                        pass

            with self.lock:
                detections = self.latest_detections
                current_annotated = self.annotated_frame

            if current_annotated is not None and (step_count % 5 == 0):
                cv2.imshow("Webots Autonomous Camera (YOLO Multi-Class View)", current_annotated)
                cv2.waitKey(1)

            pedestrian_detected = False
            is_rss_safe = True
            
            if detections:
                # クラスIDが人間であり、かつ信頼度が 0.55 以上（ガードレールの誤認識対策）のものに絞り込む
                pedestrian_detections = [
                    d for d in detections 
                    if d['class_id'] == CLASS_ID_PERSON and d.get('confidence', d.get('conf', 1.0)) >= 0.55
                ]
                
                if pedestrian_detections:
                    pedestrian_detected = True
                    closest = max(pedestrian_detections, key=lambda d: (d['bbox'][3] - d['bbox'][1]))
                    distance = self._estimate_pedestrian_distance(closest['bbox'])

                    rho = RSS_REACTION_TIME
                    a_accel = RSS_MAX_ACCEL
                    a_brake = RSS_MIN_BRAKE
                    v_r = max(curr_speed_ms, 0.0)
                    d_min = (v_r * rho + 0.5 * a_accel * (rho ** 2) + ((v_r + rho * a_accel) ** 2) / (2.0 * a_brake))

                    if distance is not None:
                        is_safe = (distance >= d_min)
                    else:
                        is_safe = False
                    is_rss_safe = is_safe

            # ==========================================================
            # 【修正】ゴール到着ロジックの全面書き直し
            #
            # 旧コードの `if curr_dist < 2.0: if curr_dist < 0.0:` は、
            # curr_dist（残り距離）が物理的に負の値を取ることが無いため
            # 内側の条件が絶対にTrueにならず、停止保持ブロックが
            # 一度も実行されない死んだコードになっていた。
            # → ゴール到着後もSTATE_CRUISEのまま40km/h目標が出続け、
            #   車がゴールを素通りしてしまうバグの直接の原因。
            #
            # 修正方針:
            #   1. GOAL_DECEL_START_DIST_M を切ったら距離に比例して
            #      目標速度を0まで線形に落とす（滑らかな減速）。
            #   2. GOAL_STOP_DIST_M 以内に入ったら STATE_ARRIVED に
            #      「固定」する（curr_distが多少前後しても後戻りしない）。
            #   3. STATE_ARRIVED では速度0・操舵0を維持し続けるだけで、
            #      while文をbreakしない = シミュレーションは継続する。
            # ==========================================================
            if curr_dist <= GOAL_STOP_DIST_M:
                goal_speed_cap_kmh = 0.0
            elif curr_dist <= GOAL_DECEL_START_DIST_M:
                ratio = (curr_dist - GOAL_STOP_DIST_M) / (GOAL_DECEL_START_DIST_M - GOAL_STOP_DIST_M)
                goal_speed_cap_kmh = CRUISE_SPEED_KMH * ratio
            else:
                goal_speed_cap_kmh = None  # 制限なし（まだゴールから十分遠い）

            # 一度ゴール圏内に入ったら STATE_ARRIVED に固定し、以後は戻らない
            if self.current_state == STATE_ARRIVED or curr_dist <= GOAL_STOP_DIST_M:
                self.current_state = STATE_ARRIVED

            target_speed_kmh = CRUISE_SPEED_KMH

            if self.current_state == STATE_ARRIVED:
                target_speed_kmh = 0.0
                steering_angle = 0.0
                if not self._arrived_logged:
                    print(f"[FSM] 最終ゴールに到着しました。停止状態を維持します（シミュレーションは継続）。")
                    self._arrived_logged = True

            elif self.current_state == STATE_CRUISE:
                target_speed_kmh = 40.0
                if pedestrian_detected and (not is_rss_safe):
                    self.current_state = STATE_STOP_INTERSECTION
                    print("[FSM][WARNING] 走行中に本物の歩行者を検知！急ブレーキ・停止フェーズに移行します")
                elif self.trigger_manager.should_trigger_stop(self.lateral.current_wp_idx):
                    self.current_state = STATE_STOP_INTERSECTION
                    print("[FSM] 交差点エリア突入: 減速・停止フェーズに移行します（自動検出トリガー）")

            elif self.current_state == STATE_STOP_INTERSECTION:
                target_speed_kmh = 0.0
                if (curr_speed_ms * 3.6) < 0.5:
                    self.current_state = STATE_WAITING
                    self.wait_timer = 0.0
                    self.safe_consecutive_count = 0
                    print("[FSM] 停止完了: 安全確認中...")

            elif self.current_state == STATE_WAITING:
                target_speed_kmh = 0.0
                self.wait_timer += self.TIME_STEP / 1000.0

                if (self.wait_timer >= MIN_MANDATORY_WAIT_SEC) and (not pedestrian_detected) and is_rss_safe:
                    self.safe_consecutive_count += 1
                else:
                    self.safe_consecutive_count = 0

                if self.safe_consecutive_count >= self.REQUIRED_SAFE_COUNT:
                    self.current_state = STATE_RE_ACCELERATE
                    print(f"[FSM] 規定待機時間経過 ＆ 安全確認OK: 再加速します！")
                elif self.wait_timer >= STATE_WAITING_FALLBACK_TIMEOUT_SEC:
                    self.current_state = STATE_RE_ACCELERATE
                    print(f"[FSM][WARNING] フォールバックにより強制発進します。")

            elif self.current_state == STATE_RE_ACCELERATE:
                target_speed_kmh = 40.0
                if pedestrian_detected and (not is_rss_safe):
                    target_speed_kmh = 0.0
                    print("[FSM][WARNING] 再加速中に歩行者を検知！緊急停止します")
                elif (curr_speed_ms * 3.6) >= RE_ACCEL_RESUME_CRUISE_SPEED_KMH:
                    self.current_state = STATE_CRUISE
                    print("[FSM] 巡航状態へ復帰します")

            # ゴール接近による速度上限を、FSMが出した目標速度の上に重ねて適用する。
            # これにより STATE_CRUISE / STATE_RE_ACCELERATE 中であっても
            # ゴール手前 GOAL_DECEL_START_DIST_M から滑らかに減速する。
            if goal_speed_cap_kmh is not None:
                target_speed_kmh = min(target_speed_kmh, goal_speed_cap_kmh)

            self._apply_actuation(target_speed_kmh, steering_angle)

            if step_count % LOG_EVERY_N_STEPS == 0:
                print(
                    f"[{step_count:05d}] State:{self.current_state} pos=({curr_x:.1f},{curr_y:.1f}) "
                    f"v={curr_speed_ms*3.6:4.1f}km/h (Target:{target_speed_kmh:4.1f}km/h) "
                    f"steer={steering_angle:+.3f} rem_dist={curr_dist:5.1f}m"
                )

        self.inference_running = False
        cv2.destroyAllWindows()
        print("[FSM] シミュレーションが終了しました。")


if __name__ == "__main__":
    print("[GUI] 道路ネットワークマップを起動しています。ゴールノードをクリックして選択してください...")
    selector = InteractiveRouteSelector(ROAD_GRAPH, default_start=DEFAULT_START_NODE_ID, default_goal=DEFAULT_GOAL_NODE_ID)
    chosen_goal = selector.show()
    print(f"[GUI] 選択されたゴールノード: {chosen_goal}")

    # 選択・確定されたスタートノードとゴールノードをコントローラに渡す
    controller = IntegratedAutonomousController(
        start_node_id=selector.start_node,  
        goal_node_id=chosen_goal,
    )
    controller.run()