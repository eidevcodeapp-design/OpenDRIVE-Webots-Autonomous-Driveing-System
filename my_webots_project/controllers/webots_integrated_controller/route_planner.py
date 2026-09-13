"""
route_planner.py
Webots自動運転コントローラ用 経路探索・ウェイポイント自動生成モジュール

- 道路網をグラフ構造（ノード=交差点/地点、エッジ=道路）として定義
- ダイクストラ法で始点から目的地までの最短経路（ノード列）を探索
- 経路上の座標列から、Pure Pursuit用の高密度ウェイポイントを自動生成する
  ための「粗いウェイポイント」を組み立てる（密度化自体は既存のdensify_pathに任せる）
- 経路上の交差点・停止線を検出し、FSM用の減速・停止トリガーを自動抽出する
"""

import heapq
import math
import numpy as np


# ============================================================
# 1. 道路網データ（グラフ構造）
#    ノード: 交差点や経由点。type / stop フラグで意味づけする
#    エッジ: ノード間の道路。中間形状点(shape_points)があればカーブ形状を表現できる
#
#    ※座標は元コードのRAW_WAYPOINTSをそのままノード化したもの。
#      新しい道路を増やしたい場合は nodes / edges に追記するだけでよい。
# ============================================================
ROAD_GRAPH = {
    "nodes": {
        "N1": {"pos": (-97.0, -52.4), "type": "start"},
        "N2": {"pos": (-97.5, -21.0), "type": "waypoint"},
        "N3": {"pos": (-96.0,   9.27), "type": "waypoint"},
        "N4": {"pos": (-92.8,   18.6),  "type": "waypoint"},
        "N5": {"pos": (-79.9,   32.9),  "type": "intersection", "stop": True}, # 死角交差点と仮定
        "N6": {"pos": (-72.3,   36.4),  "type": "waypoint"},
        "N7": {"pos": (-47.8,   37.5),  "type": "waypoint"},
        "N8": {"pos": (-20.3,   37.4),  "type": "goal"},
        
        # --- 新しく追加するノード ---
        # ① 死角交差点を直進した先のビル横 (-28.4, 14.0)
        "N9": {"pos": (-28.4, 14.0), "type": "goal"},
        
        # ② 交差点を左折したあと、おしゃれな建物の横へ向かう経由点やゴール
        "N10": {"pos": (-79.9, 14.0), "type": "waypoint"}, # 左折後の調整用など
        "N11": {"pos": (29.8, -61.2), "type": "goal"},    # おしゃれな建物の横
    },
    "edges": [
        {"from": "N1", "to": "N2"},
        {"from": "N2", "to": "N3"},
        {"from": "N3", "to": "N4"},
        {"from": "N4", "to": "N5"},
        {"from": "N5", "to": "N6"},
        {"from": "N6", "to": "N7"},
        {"from": "N7", "to": "N8"},
        
        # --- 新しく追加するエッジ（道路のつながり） ---
        # 例: N5（交差点）から直進してビル横 (N9) へ行くルート
        # （途中にウェイポイントを挟みたい場合は N5 -> N? -> N9 のように繋ぎます）
        {"from": "N5", "to": "N9"},
        
        # 例: N5（交差点）から左折して、おしゃれな建物 (N11) へ行くルート
        {"from": "N5", "to": "N10"},
        {"from": "N10", "to": "N11"},
    ],
}

# ============================================================
# 2. ダイクストラ法によるルート探索
# ============================================================
def find_route(graph, start_id, goal_id):
    """
    graph: ROAD_GRAPH形式の辞書
    return: (node_id_list, edge_list) のタプル。ルートが無ければ None
    """
    nodes = graph["nodes"]
    adjacency = {n: [] for n in nodes}

    for edge in graph["edges"]:
        x1, y1 = nodes[edge["from"]]["pos"]
        x2, y2 = nodes[edge["to"]]["pos"]
        cost = edge.get("cost", math.hypot(x2 - x1, y2 - y1))

        adjacency[edge["from"]].append((edge["to"], cost, edge))
        if edge.get("bidirectional", True):
            adjacency[edge["to"]].append((edge["from"], cost, edge))

    dist = {n: math.inf for n in nodes}
    dist[start_id] = 0.0
    prev_node = {}
    prev_edge = {}
    visited = set()
    pq = [(0.0, start_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal_id:
            break
        for v, w, edge in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = edge
                heapq.heappush(pq, (nd, v))

    if goal_id != start_id and goal_id not in prev_node:
        return None  # ルートなし

    node_path = [goal_id]
    edge_path = []
    cur = goal_id
    while cur != start_id:
        edge_path.append(prev_edge[cur])
        cur = prev_node[cur]
        node_path.append(cur)
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


# ============================================================
# 3. 経路 → 粗ウェイポイント（既存のdensify_path関数への入力を作る）
# ============================================================
def route_to_raw_waypoints(graph, node_path, edge_path):
    """
    ノード列＋エッジ列から、粗いウェイポイント配列(np.array)を作る。
    エッジに中間形状点 "shape_points" があればカーブとして反映する。
    """
    nodes = graph["nodes"]
    coords = [nodes[node_path[0]]["pos"]]

    for node_id, edge in zip(node_path[1:], edge_path):
        shape_points = edge.get("shape_points", [])
        coords.extend(shape_points)
        coords.append(nodes[node_id]["pos"])

    return np.array(coords, dtype=float)


# ============================================================
# 4. 経路上の交差点・停止線 → FSM用トリガー座標を自動抽出
# ============================================================
def extract_stop_trigger_positions(graph, node_path):
    """
    ノード列の中から type=="intersection" または stop=True のノードを
    停止トリガー地点として抽出する。
    """
    nodes = graph["nodes"]
    positions = []
    for node_id in node_path:
        node = nodes[node_id]
        if node.get("type") == "intersection" or node.get("stop"):
            positions.append(node["pos"])
    return positions


# ============================================================
# 5. トリガー管理クラス
#    「密なウェイポイント上のどのインデックスが交差点に対応するか」を
#    事前に求めておき、走行中は残距離ベースでトリガー判定を行う。
#    LateralController.remaining_distances（各ウェイポイントからゴールまでの
#    残距離配列）をそのまま流用できるように設計している。
# ============================================================
class RouteTriggerManager:
    def __init__(self, dense_waypoints, remaining_distances, trigger_positions, trigger_margin=8.0):
        """
        dense_waypoints:     LateralControllerに渡した高密度ウェイポイント配列
        remaining_distances: LateralController.remaining_distances と同一の配列
        trigger_positions:   extract_stop_trigger_positions() の出力
        trigger_margin:      交差点の何m手前でトリガーを発火させるか
        """
        self.remaining_distances = remaining_distances
        self.trigger_margin = trigger_margin

        self.trigger_indices = []
        for pos in trigger_positions:
            dists = np.hypot(dense_waypoints[:, 0] - pos[0], dense_waypoints[:, 1] - pos[1])
            self.trigger_indices.append(int(np.argmin(dists)))

        self.triggered_flags = [False] * len(self.trigger_indices)

    def should_trigger_stop(self, current_wp_idx):
        """
        現在のウェイポイントindexを渡すと、
        まだ発火していないトリガーのうち「margin以内に近づいたもの」があれば
        Trueを返し、そのトリガーを発火済みにする。
        """
        for i, trigger_idx in enumerate(self.trigger_indices):
            if self.triggered_flags[i]:
                continue
            if trigger_idx <= current_wp_idx:
                # 取りこぼし防止：既に通過している場合は発火済み扱いにする
                self.triggered_flags[i] = True
                continue

            dist_to_trigger = (
                self.remaining_distances[current_wp_idx] - self.remaining_distances[trigger_idx]
            )
            if dist_to_trigger <= self.trigger_margin:
                self.triggered_flags[i] = True
                return True
        return False

    def has_remaining_triggers(self):
        return any(not flag for flag in self.triggered_flags)

    def reset(self):
        self.triggered_flags = [False] * len(self.trigger_indices)


# ============================================================
# 動作確認用（このファイル単体でテストできる: python route_planner.py）
# ============================================================
if __name__ == "__main__":
    result = find_route(ROAD_GRAPH, "N1", "N8")
    if result is None:
        print("ルートが見つかりません")
    else:
        node_path, edge_path = result
        print("経由ノード:", node_path)

        raw_wp = route_to_raw_waypoints(ROAD_GRAPH, node_path, edge_path)
        print("粗ウェイポイント数:", len(raw_wp))
        print(raw_wp)

        stop_positions = extract_stop_trigger_positions(ROAD_GRAPH, node_path)
        print("停止トリガー地点:", stop_positions)