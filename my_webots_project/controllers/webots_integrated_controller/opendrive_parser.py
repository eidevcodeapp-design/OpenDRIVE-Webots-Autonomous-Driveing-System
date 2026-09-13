"""
opendrive_parser.py
Webots自動運転コントローラ用 - OpenDRIVE(.xodr) HDマップパーサ [ステップ1 / 都市スケール拡張版]

概要
----
OpenDRIVE形式のXML（<OpenDRIVE><road>...）を読み込み、道路同士の
接続関係(predecessor/successor, junction)・形状(planView/geometry)・
車線構成(lanes)・信号/標識(signals)を解析して、既存の route_planner.ROAD_GRAPH
と完全互換な { "nodes": {...}, "edges": [...] } 形式のグラフ構造を
座標のハードコーディング無しに自動構築する。

出力フォーマットが route_planner.ROAD_GRAPH と同一であるため、
    - find_route()
    - route_to_raw_waypoints()
    - extract_stop_trigger_positions()
    - RouteTriggerManager
    - map_gui.CityMapGUI ( / webots_integrated_controller.InteractiveRouteSelector )
は一切変更せずにそのまま使い回せる「ドロップイン差し替え」構造になっている。

対応ジオメトリ (<planView><geometry>):
    - <line/>                          直線
    - <arc curvature="k"/>             円弧（定曲率）
    - <spiral curvStart= curvEnd=/>    クロソイド（曲率が線形に変化するカーブ）
                                       → heading/位置を数値積分（台形則）して近似
    - <paramPoly3 aU bU ... pRange=/>  3次多項式カーブ（実xodrで最頻出のカーブ形状）
未対応（今後の拡張ポイント）:
    - <poly3/>（廃止済み旧仕様。paramPoly3への変換が公式にも推奨されている）

ノード生成ロジック:
    各道路(road)の始点・終点をエンドポイントとして列挙し、座標が
    snap_tolerance 以内で近接するエンドポイント同士を Union-Find で
    クラスタリングして1つのノードにまとめる（= 道路網の「継ぎ目」を自動検出）。
    近接判定は scipy.spatial.cKDTree があれば query_pairs() でO(n log n)、
    無ければ従来のO(n^2)総当たりにフォールバックする（数千道路規模の街でも
    実用的な速度で処理できるようにするための拡張）。
    3本以上の道路が集まるノード、または <junction> に属する道路が接続する
    ノードは type="intersection" として扱う。

一方通行の自動判定:
    <lanes><laneSection><left>/<right> 内の type="driving" レーン数を数え、
    片側にしか走行レーンが無い道路は一方通行として扱う
    （right側のみ → s増加方向のみ通行可、left側のみ → s減少方向のみ通行可、
    これはOpenDRIVEの標準的なレーン方向の約束事に基づく）。
    両側にレーンがある、または<lanes>情報が無い道路は従来通り双方向として扱う
    （後方互換：生成済みの route_planner.ROAD_GRAPH 形式データはそのまま動く）。

停止トリガー判定:
    <signals><signal .../> のうち、type が STOP_SIGNAL_TYPES に含まれる、
    または name に "stop" を含むものを検出し、最寄りのノードに stop=True を
    付与する（実データでは自国のOpenDRIVE信号カタログに合わせて
    STOP_SIGNAL_TYPES を調整すること。例: ドイツSTVOカタログでは "206" が
    一時停止標識）。
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
    SCIPY_SPATIAL_AVAILABLE = True
except ImportError:
    SCIPY_SPATIAL_AVAILABLE = False


# ============================================================
# 設定値
# ============================================================
STOP_SIGNAL_TYPES = {"206", "294"}    # 停止標識とみなす信号typeコード（要調整）
DEFAULT_SAMPLE_STEP = 2.0            # 道路形状のサンプリング間隔[m]
DEFAULT_SNAP_TOLERANCE = 0.75        # 同一地点とみなす座標の許容誤差[m]


# ============================================================
# 1. データモデル
# ============================================================
@dataclass
class GeometrySegment:
    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str                     # "line" | "arc" | "spiral" | "param_poly3" | "unsupported"
    curvature: float = 0.0
    # --- spiral 用 ---
    curv_start: float = 0.0
    curv_end: float = 0.0
    # --- paramPoly3 用 ---
    au: float = 0.0
    bu: float = 0.0
    cu: float = 0.0
    du: float = 0.0
    av: float = 0.0
    bv: float = 0.0
    cv: float = 0.0
    dv: float = 0.0
    p_range: str = "normalized"    # "normalized" | "arcLength"


@dataclass
class SignalInfo:
    s: float
    signal_type: str
    name: str


@dataclass
class RoadLink:
    element_type: Optional[str] = None    # "road" | "junction"
    element_id: Optional[str] = None
    contact_point: Optional[str] = None   # "start" | "end"


@dataclass
class LaneSectionInfo:
    s: float
    left_driving: int = 0
    right_driving: int = 0


@dataclass
class Road:
    road_id: str
    length: float
    junction: str                         # "-1" ならジャンクション非所属
    geometries: list = field(default_factory=list)
    predecessor: Optional[RoadLink] = None
    successor: Optional[RoadLink] = None
    signals: list = field(default_factory=list)
    lane_sections: list = field(default_factory=list)


@dataclass
class OpenDriveMap:
    roads: dict = field(default_factory=dict)    # road_id -> Road


# ============================================================
# 2. XMLパース
# ============================================================
def parse_opendrive(source, from_string: bool = False) -> OpenDriveMap:
    """
    source: .xodrファイルのパス。from_string=True の場合はXML文字列そのもの。
    """
    root = ET.fromstring(source) if from_string else ET.parse(source).getroot()
    odr_map = OpenDriveMap()

    for road_elem in root.findall("road"):
        road_id = road_elem.get("id")
        length = float(road_elem.get("length", "0"))
        junction = road_elem.get("junction", "-1")
        road = Road(road_id=road_id, length=length, junction=junction)

        # --- link (predecessor / successor) ---
        link_elem = road_elem.find("link")
        if link_elem is not None:
            pred = link_elem.find("predecessor")
            if pred is not None:
                road.predecessor = RoadLink(
                    element_type=pred.get("elementType"),
                    element_id=pred.get("elementId"),
                    contact_point=pred.get("contactPoint"),
                )
            succ = link_elem.find("successor")
            if succ is not None:
                road.successor = RoadLink(
                    element_type=succ.get("elementType"),
                    element_id=succ.get("elementId"),
                    contact_point=succ.get("contactPoint"),
                )

        # --- planView / geometry ---
        plan_view = road_elem.find("planView")
        if plan_view is not None:
            for geom_elem in plan_view.findall("geometry"):
                s = float(geom_elem.get("s", "0"))
                x = float(geom_elem.get("x", "0"))
                y = float(geom_elem.get("y", "0"))
                hdg = float(geom_elem.get("hdg", "0"))
                glen = float(geom_elem.get("length", "0"))

                if geom_elem.find("line") is not None:
                    seg = GeometrySegment(s, x, y, hdg, glen, kind="line")

                elif geom_elem.find("arc") is not None:
                    curvature = float(geom_elem.find("arc").get("curvature", "0"))
                    seg = GeometrySegment(s, x, y, hdg, glen, kind="arc", curvature=curvature)

                elif geom_elem.find("spiral") is not None:
                    sp = geom_elem.find("spiral")
                    seg = GeometrySegment(
                        s, x, y, hdg, glen, kind="spiral",
                        curv_start=float(sp.get("curvStart", "0")),
                        curv_end=float(sp.get("curvEnd", "0")),
                    )

                elif geom_elem.find("paramPoly3") is not None:
                    pp = geom_elem.find("paramPoly3")
                    seg = GeometrySegment(
                        s, x, y, hdg, glen, kind="param_poly3",
                        au=float(pp.get("aU", "0")), bu=float(pp.get("bU", "0")),
                        cu=float(pp.get("cU", "0")), du=float(pp.get("dU", "0")),
                        av=float(pp.get("aV", "0")), bv=float(pp.get("bV", "0")),
                        cv=float(pp.get("cV", "0")), dv=float(pp.get("dV", "0")),
                        p_range=pp.get("pRange", "normalized"),
                    )

                else:
                    print(
                        f"[opendrive_parser][WARNING] road={road_id} s={s}: "
                        f"未対応ジオメトリ(poly3等)のため直線近似します"
                    )
                    seg = GeometrySegment(s, x, y, hdg, glen, kind="unsupported")

                road.geometries.append(seg)

        # --- signals（停止標識・信号機） ---
        signals_elem = road_elem.find("signals")
        if signals_elem is not None:
            for sig_elem in signals_elem.findall("signal"):
                road.signals.append(SignalInfo(
                    s=float(sig_elem.get("s", "0")),
                    signal_type=str(sig_elem.get("type", "")),
                    name=sig_elem.get("name", ""),
                ))

        # --- lanes（一方通行判定用） ---
        lanes_elem = road_elem.find("lanes")
        if lanes_elem is not None:
            for ls_elem in lanes_elem.findall("laneSection"):
                ls = LaneSectionInfo(s=float(ls_elem.get("s", "0")))
                left_elem = ls_elem.find("left")
                if left_elem is not None:
                    ls.left_driving = sum(
                        1 for lane in left_elem.findall("lane")
                        if lane.get("type") == "driving"
                    )
                right_elem = ls_elem.find("right")
                if right_elem is not None:
                    ls.right_driving = sum(
                        1 for lane in right_elem.findall("lane")
                        if lane.get("type") == "driving"
                    )
                road.lane_sections.append(ls)

        odr_map.roads[road_id] = road

    return odr_map


# ============================================================
# 3. ジオメトリ -> 座標列のサンプリング
# ============================================================
def _sample_line_or_arc(seg: GeometrySegment, step: float) -> np.ndarray:
    n = max(int(math.ceil(seg.length / step)), 1)
    s_values = np.linspace(0.0, seg.length, n + 1)

    if seg.kind == "arc" and abs(seg.curvature) > 1e-9:
        k = seg.curvature
        hdg = seg.hdg + k * s_values
        x = seg.x + (np.sin(hdg) - math.sin(seg.hdg)) / k
        y = seg.y - (np.cos(hdg) - math.cos(seg.hdg)) / k
    else:
        x = seg.x + s_values * math.cos(seg.hdg)
        y = seg.y + s_values * math.sin(seg.hdg)

    return np.column_stack([x, y])


def _sample_spiral(seg: GeometrySegment, step: float) -> np.ndarray:
    if seg.length <= 0:
        return np.array([[seg.x, seg.y]])

    fine_step = min(step, 0.5)
    n = max(int(math.ceil(seg.length / fine_step)), 2)
    s_values = np.linspace(0.0, seg.length, n + 1)

    dk = (seg.curv_end - seg.curv_start) / seg.length
    heading = seg.hdg + seg.curv_start * s_values + 0.5 * dk * (s_values ** 2)

    ds = np.diff(s_values)
    dx = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(heading[:-1]) + np.cos(heading[1:])) * ds)])
    dy = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(heading[:-1]) + np.sin(heading[1:])) * ds)])

    x = seg.x + dx
    y = seg.y + dy
    return np.column_stack([x, y])


def _sample_param_poly3(seg: GeometrySegment, step: float) -> np.ndarray:
    if seg.length <= 0:
        return np.array([[seg.x, seg.y]])

    n = max(int(math.ceil(seg.length / step)), 1)
    s_values = np.linspace(0.0, seg.length, n + 1)

    if seg.p_range == "arcLength":
        p = s_values
    else:
        p = s_values / seg.length

    u = seg.au + seg.bu * p + seg.cu * p ** 2 + seg.du * p ** 3
    v = seg.av + seg.bv * p + seg.cv * p ** 2 + seg.dv * p ** 3

    cos_h, sin_h = math.cos(seg.hdg), math.sin(seg.hdg)
    x = seg.x + u * cos_h - v * sin_h
    y = seg.y + u * sin_h + v * cos_h
    return np.column_stack([x, y])


def _sample_segment(seg: GeometrySegment, step: float) -> np.ndarray:
    if seg.length <= 0:
        return np.array([[seg.x, seg.y]])

    if seg.kind == "spiral":
        return _sample_spiral(seg, step)
    if seg.kind == "param_poly3":
        return _sample_param_poly3(seg, step)
    return _sample_line_or_arc(seg, step)


def sample_road(road: Road, step: float = DEFAULT_SAMPLE_STEP) -> np.ndarray:
    chunks = []
    for i, seg in enumerate(sorted(road.geometries, key=lambda g: g.s)):
        pts = _sample_segment(seg, step)
        if i > 0 and len(chunks) > 0:
            pts = pts[1:]
        chunks.append(pts)
    return np.vstack(chunks) if chunks else np.zeros((0, 2))


# ============================================================
# 4. Union-Find（エンドポイントのクラスタリング用）
# ============================================================
class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


def _cluster_endpoints(endpoints, odr_map, snap_tolerance, link_sanity_multiplier=20.0):
    n = len(endpoints)
    uf = _UnionFind(n)
    if n <= 1:
        return uf

    index_of = {(rid, which): i for i, (rid, which, x, y) in enumerate(endpoints)}

    for rid, road in odr_map.roads.items():
        for link, which in ((road.predecessor, "start"), (road.successor, "end")):
            if link is None or link.element_type != "road":
                continue
            this_key = (rid, which)
            other_key = (link.element_id, link.contact_point)
            if this_key not in index_of or other_key not in index_of:
                continue

            i, j = index_of[this_key], index_of[other_key]
            xi, yi = endpoints[i][2], endpoints[i][3]
            xj, yj = endpoints[j][2], endpoints[j][3]
            dist = math.hypot(xi - xj, yi - yj)

            if dist <= snap_tolerance * link_sanity_multiplier:
                uf.union(i, j)
            else:
                print(
                    f"[opendrive_parser][WARNING] road={rid} の{which}端と "
                    f"road={link.element_id}({link.contact_point}) 端が {dist:.2f}m "
                    f"も離れています。"
                )

    if SCIPY_SPATIAL_AVAILABLE:
        coords = np.array([[e[2], e[3]] for e in endpoints])
        tree = cKDTree(coords)
        for i, j in tree.query_pairs(r=snap_tolerance):
            uf.union(i, j)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if math.hypot(endpoints[i][2] - endpoints[j][2],
                            endpoints[i][3] - endpoints[j][3]) <= snap_tolerance:
                    uf.union(i, j)
    return uf


# ============================================================
# 5. グラフ構築
# ============================================================
def build_road_graph(
    odr_map: OpenDriveMap,
    step: float = DEFAULT_SAMPLE_STEP,
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE,
    start_road_id: Optional[str] = None,
    goal_road_id: Optional[str] = None,
    use_lane_direction: bool = True,
):
    road_samples = {rid: sample_road(r, step) for rid, r in odr_map.roads.items()}

    endpoints = []
    for rid, pts in road_samples.items():
        if len(pts) == 0:
            continue
        endpoints.append((rid, "start", pts[0][0], pts[0][1]))
        endpoints.append((rid, "end", pts[-1][0], pts[-1][1]))

    uf = _cluster_endpoints(endpoints, odr_map, snap_tolerance)

    cluster_to_node = {}
    node_id_of = {}
    nodes = {}
    for idx in range(len(endpoints)):
        root = uf.find(idx)
        if root not in cluster_to_node:
            node_id = f"J{len(cluster_to_node)}"
            cluster_to_node[root] = node_id
            nodes[node_id] = {"pos": None, "type": "waypoint", "stop": False, "_members": []}
        node_id = cluster_to_node[root]
        node_id_of[idx] = node_id
        nodes[node_id]["_members"].append(endpoints[idx])

    # --- クラスタの代表座標(重心)とtype判定 ---
    for node_id, ndata in nodes.items():
        members = ndata["_members"]
        cx = sum(m[2] for m in members) / len(members)
        cy = sum(m[3] for m in members) / len(members)
        ndata["pos"] = (cx, cy)

        involved_roads = {m[0] for m in members}
        touches_junction = any(odr_map.roads[rid].junction != "-1" for rid in involved_roads)
        
        # 【対策Aの適用】3本以上の接続判定を廃止し、ジャンクションまたはstopフラグがある場合のみ交差点とする
        if touches_junction or ndata.get("stop", False):
            ndata["type"] = "intersection"
            
        del ndata["_members"]

    def _endpoint_node(road_id, which):
        for idx, ep in enumerate(endpoints):
            if ep[0] == road_id and ep[1] == which:
                return node_id_of[idx]
        return None

    if start_road_id is not None:
        n = _endpoint_node(start_road_id, "start")
        if n:
            nodes[n]["type"] = "start"
    if goal_road_id is not None:
        n = _endpoint_node(goal_road_id, "end")
        if n:
            nodes[n]["type"] = "goal"

    for rid, road in odr_map.roads.items():
        pts = road_samples.get(rid)
        if pts is None or len(pts) == 0:
            continue
        sample_s = np.linspace(0.0, road.length, len(pts))

        for sig in road.signals:
            is_stop = (sig.signal_type in STOP_SIGNAL_TYPES) or ("stop" in sig.name.lower())
            if not is_stop:
                continue
            nearest_pt_idx = int(np.argmin(np.abs(sample_s - sig.s)))
            sx, sy = pts[nearest_pt_idx]

            nearest_node, nearest_dist = None, math.inf
            for node_id, ndata in nodes.items():
                nx, ny = ndata["pos"]
                d = math.hypot(nx - sx, ny - sy)
                if d < nearest_dist:
                    nearest_dist, nearest_node = d, node_id
            if nearest_node is not None:
                nodes[nearest_node]["stop"] = True
                if nodes[nearest_node]["type"] == "waypoint":
                    nodes[nearest_node]["type"] = "intersection"

    edges = []
    for rid, pts in road_samples.items():
        if len(pts) < 2:
            continue
        road = odr_map.roads[rid]
        from_node = _endpoint_node(rid, "start")
        to_node = _endpoint_node(rid, "end")
        mid_points = [tuple(p) for p in pts[1:-1]]

        bidirectional = True
        lane_count = 2
        if use_lane_direction and road.lane_sections:
            ls = min(road.lane_sections, key=lambda s: s.s)
            left_n, right_n = ls.left_driving, ls.right_driving
            lane_count = max(left_n + right_n, 1)

            if right_n > 0 and left_n == 0:
                bidirectional = False
            elif left_n > 0 and right_n == 0:
                from_node, to_node = to_node, from_node
                mid_points = list(reversed(mid_points))
                bidirectional = False

        edges.append({
            "from": from_node,
            "to": to_node,
            "shape_points": mid_points,
            "bidirectional": bidirectional,
            "lane_count": lane_count,
            "road_id": rid,
        })

    return {"nodes": nodes, "edges": edges}


def load_road_graph_from_xodr(path: str, **build_kwargs):
    odr_map = parse_opendrive(path, from_string=False)
    return build_road_graph(odr_map, **build_kwargs)


# ============================================================
# 6. 動作確認用サンプルOpenDRIVE
# ============================================================
def generate_sample_xodr() -> str:
    raw_points = [
        (-97.0, -52.4), (-97.5, -21.0), (-96.0, 9.27), (-92.8, 18.6),
        (-79.9, 32.9), (-72.3, 36.4), (-47.8, 37.5), (-20.3, 37.4),
    ]

    roads_xml = []
    for i in range(7):
        x0, y0 = raw_points[i]
        x1, y1 = raw_points[i + 1]
        hdg = math.atan2(y1 - y0, x1 - x0)
        length = math.hypot(x1 - x0, y1 - y0)
        road_id = str(i + 1)

        pred = (f'<predecessor elementType="road" elementId="{str(i)}" contactPoint="end"/>' if i > 0 else "")
        succ = (f'<successor elementType="road" elementId="{str(i + 2)}" contactPoint="start"/>' if i < 6 else "")

        signal_xml = ""
        if i == 3:
            signal_xml = (f'<signals><signal s="{length:.3f}" t="0" '
                          f'name="Stop_N5" type="206" dynamic="no"/></signals>')

        roads_xml.append(f'''
    <road name="road_{road_id}" length="{length:.3f}" id="{road_id}" junction="-1">
      <link>{pred}{succ}</link>
      <planView>
        <geometry s="0" x="{x0}" y="{y0}" hdg="{hdg:.6f}" length="{length:.3f}">
          <line/>
        </geometry>
      </planView>
      {signal_xml}
    </road>''')

    branch_roads = [
        ("8", (-47.8, 37.5), (-40.4, 37.0), "6", "9"),
        ("9", (-40.4, 37.0), (-33.0, 36.9), "8", "10"),
        ("10", (-33.0, 36.9), (-29.8, 31.3), "9", "11"),
        ("11", (-29.8, 31.3), (-28.4, 14.0), "10", None),
    ]
    
    for item in branch_roads:
        r_id, p0, p1, prev_r, next_r = item[0], item[1], item[2], item[3], item[4]
        x0, y0 = p0
        x1, y1 = p1
        hdg = math.atan2(y1 - y0, x1 - x0)
        length = math.hypot(x1 - x0, y1 - y0)
        
        pred_xml = f'<predecessor elementType="road" elementId="{prev_r}" contactPoint="end"/>' if prev_r else ""
        succ_xml = f'<successor elementType="road" elementId="{next_r}" contactPoint="start"/>' if next_r else ""
        
        roads_xml.append(f'''
    <road name="road_{r_id}" length="{length:.3f}" id="{r_id}" junction="-1">
      <link>{pred_xml}{succ_xml}</link>
      <planView>
        <geometry s="0" x="{x0}" y="{y0}" hdg="{hdg:.6f}" length="{length:.3f}">
          <line/>
        </geometry>
      </planView>
    </road>''')

    return f'<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE>{"".join(roads_xml)}\n</OpenDRIVE>'


def _lanes_xml(n_right=1, n_left=1):
    right_lanes = "".join(
        f'<lane id="-{i+1}" type="driving" level="false">'
        f'<width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>'
        for i in range(n_right)
    ) if n_right > 0 else ""
    left_lanes = "".join(
        f'<lane id="{i+1}" type="driving" level="false">'
        f'<width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>'
        for i in range(n_left)
    ) if n_left > 0 else ""
    return (
        '<lanes><laneSection s="0">'
        f'<left>{left_lanes}</left>'
        '<center><lane id="0" type="none" level="false"/></center>'
        f'<right>{right_lanes}</right>'
        '</laneSection></lanes>'
    )


def generate_sample_city_xodr() -> str:
    A, B, C, D, E = (0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0), (160.0, 80.0)

    def _road(road_id, p0, p1, pred=None, succ=None, lanes_xml="", signal_xml=""):
        x0, y0 = p0
        x1, y1 = p1
        hdg = math.atan2(y1 - y0, x1 - x0)
        length = math.hypot(x1 - x0, y1 - y0)
        pred_xml = (f'<predecessor elementType="road" elementId="{pred}" contactPoint="end"/>'
                    if pred else "")
        succ_xml = (f'<successor elementType="road" elementId="{succ}" contactPoint="start"/>'
                    if succ else "")
        return f'''
    <road name="road_{road_id}" length="{length:.3f}" id="{road_id}" junction="-1">
      <link>{pred_xml}{succ_xml}</link>
      <planView>
        <geometry s="0" x="{x0}" y="{y0}" hdg="{hdg:.6f}" length="{length:.3f}"><line/></geometry>
      </planView>
      {lanes_xml}
      {signal_xml}
    </road>'''

    stop_at_B = '<signals><signal s="95" t="0" name="Stop_B" type="206" dynamic="no"/></signals>'
    stop_at_C = '<signals><signal s="5" t="0" name="Stop_C" type="206" dynamic="no"/></signals>'

    roads_xml = [
        _road("1", A, B, lanes_xml=_lanes_xml(n_right=1, n_left=0), signal_xml=stop_at_B),
        _road("2", B, C, lanes_xml=_lanes_xml(n_right=1, n_left=1)),
        _road("3", C, D, lanes_xml=_lanes_xml(n_right=1, n_left=1), signal_xml=stop_at_C),
        _road("4", D, A, lanes_xml=_lanes_xml(n_right=1, n_left=1)),
        _road("5", C, E, lanes_xml=_lanes_xml(n_right=1, n_left=1)),
    ]

    return f'<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE>{"".join(roads_xml)}\n</OpenDRIVE>'


if __name__ == "__main__":
    from route_planner import find_route, route_to_raw_waypoints, extract_stop_trigger_positions

    print("=== [1] 単純な一本道サンプル ===")
    xodr_text = generate_sample_xodr()
    odr_map = parse_opendrive(xodr_text, from_string=True)
    graph = build_road_graph(odr_map, start_road_id="1", goal_road_id="7")
    print(f"ノード数: {len(graph['nodes'])} / エッジ数: {len(graph['edges'])}")

    start_id = next(n for n, d in graph["nodes"].items() if d["type"] == "start")
    goal_id = next(n for n, d in graph["nodes"].items() if d["type"] == "goal")
    result = find_route(graph, start_id, goal_id)
    node_path, edge_path = result
    print("探索ルート:", " -> ".join(node_path))

    print("\n=== [2] ループ状の都市サンプル（一方通行あり） ===")
    city_xodr = generate_sample_city_xodr()
    city_map = parse_opendrive(city_xodr, from_string=True)
    city_graph = build_road_graph(city_map, start_road_id="1", goal_road_id="5")
    print(f"ノード数: {len(city_graph['nodes'])} / エッジ数: {len(city_graph['edges'])}")
    for nid, nd in city_graph["nodes"].items():
        print(f"  {nid}: pos=({nd['pos'][0]:.1f},{nd['pos'][1]:.1f}) type={nd['type']} stop={nd['stop']}")
    for e in city_graph["edges"]:
        print(f"  edge road={e['road_id']}: {e['from']} -> {e['to']} "
              f"(bidirectional={e['bidirectional']}, lanes={e['lane_count']})")

    c_start = next(n for n, d in city_graph["nodes"].items() if d["type"] == "start")
    c_goal = next(n for n, d in city_graph["nodes"].items() if d["type"] == "goal")
    c_result = find_route(city_graph, c_start, c_goal)
    if c_result is None:
        print("ルートが見つかりません")
    else:
        c_node_path, c_edge_path = c_result
        print("探索ルート:", " -> ".join(c_node_path))
        raw_wp = route_to_raw_waypoints(city_graph, c_node_path, c_edge_path)
        print("粗ウェイポイント数:", len(raw_wp))
        print("停止トリガー地点:", extract_stop_trigger_positions(city_graph, c_node_path))