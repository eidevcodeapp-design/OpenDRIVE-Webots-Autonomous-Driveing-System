"""
map_gui.py
街全体道路網 2D鳥瞰図GUI ＋ クリックによるスタート/ゴール選択・経路探索統合

- opendrive_parser.build_road_graph() が出力する route_planner.ROAD_GRAPH
  互換の {"nodes": {...}, "edges": [...]} をそのまま入力にできる
  （手動で作ったROAD_GRAPHでも動く＝完全にデータ駆動）
- ノード数・エッジ数が多い都市規模のグラフでも高速にクリック判定できるよう
  scipy.spatial.cKDTree でノード検索する（未インストール時は線形探索に
  フォールバック）
- 左クリック : ゴールノードを選択 / 右クリック : スタートノードを選択
- クリックの度に route_planner.find_route()（ダイクストラ法）で経路を
  再計算し、金色でハイライト＋一方通行は矢印表示＋総距離/推定所要時間を表示
- 「🚀 走行開始」ボタンで確定し、(start_node_id, goal_node_id) をshow()の
  戻り値として返す。そのままIntegratedAutonomousController等に渡せる。
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from route_planner import find_route

import matplotlib.image as mpimg  # 追加

# 図の初期化
fig, ax = plt.subplots(figsize=(10, 10))

# --- 【ここを追加】背景の鳥瞰図画像を読み込んでぴったり敷く ---
try:
    # 保存した画像ファイルを読み込む
    bg_img = mpimg.imread('world_top_view.png')
    
    # 【重要】OpenDRIVEの道路データの座標範囲（Xの最小/最大、Yの最小/最大）に合わせます
    # ※もし道路がはみ出したりずれたりする場合は、ここの数値（範囲）を微調整します
    x_min, x_max = -100.0, 100.0  # サンプル都市や実際のOpenDRIVEのスケールに合わせて変更
    y_min, y_max = -100.0, 100.0  
    
    # imshowで画像を背景に配置 (origin='upper' または 'lower' は画像の上下反転対策)
    ax.imshow(bg_img, extent=[x_min, x_max, y_min, y_max], origin='upper', alpha=0.8)
    
except FileNotFoundError:
    print("⚠️ 背景画像 'world_top_view.png' が見つからないため、白背景で起動します。")
# -----------------------------------------------------------

try:
    from scipy.spatial import cKDTree
    SCIPY_SPATIAL_AVAILABLE = True
except ImportError:
    SCIPY_SPATIAL_AVAILABLE = False


ASSUMED_CRUISE_SPEED_KMH = 30.0   # 所要時間概算に使う巡航速度
NODE_PICK_RADIUS_M = 15.0         # クリックでノードを拾う許容半径[m]


class CityMapGUI:
    """
    道路網グラフ(ROAD_GRAPH互換dict)を受け取り、鳥瞰図をmatplotlibで描画する。
    InteractiveRouteSelectorの後継。都市規模のグラフでも軽く動くことと、
    スタート/ゴールの両方をGUI上で選び直せることが主な拡張点。
    """

    def __init__(self, road_graph, default_start, default_goal,
                 cruise_speed_kmh=ASSUMED_CRUISE_SPEED_KMH):
        self.graph = road_graph
        self.nodes = road_graph["nodes"]
        self.edges = road_graph["edges"]
        self.start_node = default_start
        self.goal_node = default_goal
        self.confirmed = False
        self.cruise_speed_kmh = cruise_speed_kmh

        self._node_ids = list(self.nodes.keys())
        self._coords = np.array([self.nodes[n]["pos"] for n in self._node_ids])
        self._tree = cKDTree(self._coords) if (SCIPY_SPATIAL_AVAILABLE and len(self._node_ids) > 0) else None

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        ax_button = plt.axes([0.72, 0.02, 0.25, 0.06])
        self.btn_start = Button(ax_button, "🚀 走行開始", color="lightgoldenrodyellow", hovercolor="0.9")
        self.btn_start.on_clicked(self._on_submit)

        self._redraw()

    # ------------------------------------------------------------
    # ノード検索（KDTreeがあれば高速、無ければ線形探索にフォールバック）
    # ------------------------------------------------------------
    def _nearest_node(self, x, y, max_dist=NODE_PICK_RADIUS_M):
        if len(self._node_ids) == 0:
            return None
        if self._tree is not None:
            dist, idx = self._tree.query([x, y])
            return self._node_ids[idx] if dist <= max_dist else None

        best_id, best_dist = None, math.inf
        for i, nid in enumerate(self._node_ids):
            d = math.hypot(self._coords[i, 0] - x, self._coords[i, 1] - y)
            if d < best_dist:
                best_dist, best_id = d, nid
        return best_id if best_dist <= max_dist else None

    # ------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        node = self._nearest_node(event.xdata, event.ydata)
        if node is None:
            return

        if event.button == 3:  # 右クリック -> スタート変更
            if node != self.goal_node:
                self.start_node = node
                print(f"[GUI] スタートノードを変更: {self.start_node}")
                self._redraw()
        else:  # 左クリック -> ゴール変更
            if node != self.start_node:
                self.goal_node = node
                print(f"[GUI] ゴールノードを変更: {self.goal_node}")
                self._redraw()

    def _on_submit(self, event):
        print(f"[GUI] ルート確定！ スタート: {self.start_node} -> ゴール: {self.goal_node}")
        self.confirmed = True
        plt.close(self.fig)

    def show(self):
        """GUIを表示し、確定された (start_node_id, goal_node_id) を返す"""
        plt.show()
        return self.start_node, self.goal_node

    # ------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------
    @staticmethod
    def _edge_points(nodes, edge):
        pts = [nodes[edge["from"]]["pos"]]
        pts.extend(edge.get("shape_points", []))
        pts.append(nodes[edge["to"]]["pos"])
        return pts

    @staticmethod
    def _edge_width(edge):
        lanes = edge.get("lane_count", 2)
        return 1.5 + 0.6 * max(lanes, 1)

    def _draw_direction_arrow(self, xs, ys):
        mid = len(xs) // 2
        mid = max(mid, 1)
        dx, dy = xs[mid] - xs[mid - 1], ys[mid] - ys[mid - 1]
        norm = math.hypot(dx, dy) or 1.0
        mx, my = xs[mid], ys[mid]
        self.ax.annotate(
            "", xy=(mx + dx / norm * 3.5, my + dy / norm * 3.5), xytext=(mx, my),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5), zorder=2,
        )

    def _redraw(self):
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True, linestyle="--", alpha=0.4)

        # --- 道路（中心線）の描画。shape_pointsがあればカーブ形状を反映 ---
        for edge in self.edges:
            pts = self._edge_points(self.nodes, edge)
            xs, ys = zip(*pts)
            self.ax.plot(xs, ys, color="#8a8a8a", linewidth=self._edge_width(edge),
                         solid_capstyle="round", zorder=1)
            if not edge.get("bidirectional", True):
                self._draw_direction_arrow(xs, ys)

        # --- 経路探索＆ハイライト ---
        try:
            result = find_route(self.graph, self.start_node, self.goal_node)
        except Exception:
            result = None

        if result:
            node_path, edge_path = result
            total_dist = 0.0
            for edge in edge_path:
                pts = self._edge_points(self.nodes, edge)
                xs, ys = zip(*pts)
                self.ax.plot(xs, ys, color="gold", linewidth=5, zorder=3, solid_capstyle="round")
                for i in range(len(pts) - 1):
                    total_dist += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            eta_min = (total_dist / 1000.0) / max(self.cruise_speed_kmh, 1e-3) * 60.0
            route_info = f"経路距離: {total_dist:,.0f} m ／ 推定所要時間: {eta_min:.1f} 分（{self.cruise_speed_kmh:.0f} km/h想定）"
        else:
            route_info = "⚠ ルートが見つかりません（道路網が分断されている可能性があります）"

        # --- ノードの描画（waypointは控えめに、交差点/停止標識/start/goalは強調） ---
        for nid, ndata in self.nodes.items():
            x, y = ndata["pos"]
            ntype = ndata.get("type", "waypoint")
            is_stop = ndata.get("stop", False)

            if nid == self.start_node:
                self.ax.scatter(x, y, s=220, marker="^", color="limegreen", zorder=5, edgecolor="black")
                self.ax.text(x, y + 3, "START", fontsize=9, fontweight="bold", ha="center")
            elif nid == self.goal_node:
                self.ax.scatter(x, y, s=260, marker="*", color="red", zorder=5, edgecolor="black")
                self.ax.text(x, y + 3, "GOAL", fontsize=9, fontweight="bold", ha="center")
            elif is_stop:
                self.ax.scatter(x, y, s=90, marker="8", color="crimson", zorder=4, edgecolor="black")
            elif ntype == "intersection":
                self.ax.scatter(x, y, s=60, color="darkorange", zorder=4)
            else:
                self.ax.scatter(x, y, s=12, color="steelblue", alpha=0.5, zorder=2)

        self.ax.set_title(
            f"Start: {self.start_node}  →  Goal: {self.goal_node}\n"
            f"左クリック=ゴール変更 / 右クリック=スタート変更\n{route_info}",
            fontsize=10,
        )
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.fig.canvas.draw_idle()


# ============================================================
# 動作確認 (python map_gui.py) -- 表示なしでも描画ロジックだけ検証したい場合は
# MPLBACKEND=Agg python map_gui.py として実行する
# ============================================================
if __name__ == "__main__":
    from opendrive_parser import parse_opendrive, build_road_graph, generate_sample_xodr as generate_sample_city_xodr
    
    odr_map = parse_opendrive(generate_sample_city_xodr(), from_string=True)
    graph = build_road_graph(odr_map, start_road_id="1", goal_road_id="5")

    start_id = next(n for n, d in graph["nodes"].items() if d["type"] == "start")
    goal_id = next(n for n, d in graph["nodes"].items() if d["type"] == "goal")

    print("[GUI] 道路ネットワークマップを起動しています。ノードをクリックしてルートを選択してください...")
    gui = CityMapGUI(graph, default_start=start_id, default_goal=goal_id)
    chosen_start, chosen_goal = gui.show()
    print(f"[GUI] 確定: start={chosen_start} goal={chosen_goal}")