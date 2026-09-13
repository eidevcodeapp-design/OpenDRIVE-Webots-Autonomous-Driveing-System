"""
================================================================================
 npc_trigger_supervisor.py
 Phase 3: 死角交差点における歩行者飛び出しトリガー制御（Supervisor）

【役割】
  - 自車(DEF EGO_VEHICLE)と横断開始位置(CROSSING_START_POSITION)との距離を毎ステップ監視
  - TRIGGER_DISTANCE_M 以内に自車が接近したら、歩行者(DEF PEDESTRIAN_NPC)を
    「隠れ位置」から「横断終了位置」へ一定時間で直線移動させる（＝飛び出し演出）
  - 横断完了後、一定時間待ってから隠れ位置に戻し、次の実験に備える（繰り返し実験用）

【なぜPedestrian標準コントローラではなくSupervisorで動かすのか】
  Webots標準のPedestrianPROTOは trajectory フィールドに沿って自律的に往復する
  独立コントローラを持つが、これは「自車の接近タイミング」と同期しない。
  IRL学習・RSS評価では「自車がある距離まで来たら必ず飛び出す」という
  再現性の高いトリガーが必要なため、Pedestrianのcontrollerフィールドを "<none>" にし、
  translationフィールドを外部のSupervisorから直接書き換える方式にしている。

【Webots側セットアップ手順】
  1. city.wbt（または新しいワールドファイル）をテキストエディタで開く
  2. Priusノードの手前に "DEF EGO_VEHICLE " を追加する
       例: DEF EGO_VEHICLE Prius {
  3. 死角交差点の角に、遮蔽物となる建物(Solid)を配置する（下記スニペット参照）
  4. 歩行者を配置する。Webots上で Add Node からPedestrianを検索して追加するか、
     テキストで直接以下のように記述する:

       DEF PEDESTRIAN_NPC Pedestrian {
         translation 8 1.27 -14
         rotation 0 1 0 -1.5708
         name "pedestrian_npc"
         controller "<none>"
       }

     ※ translationの初期値は上記のHIDDEN_POSITIONと一致させておく
     ※ y座標(高さ)はPedestrianPROTOの初期値のまま変更しないこと(接地位置がずれる)

  5. supervisorフィールドをTRUEにした空のRobotノードをワールドに追加する:

       DEF NPC_TRIGGER Robot {
         name "npc_trigger"
         controller "npc_trigger_supervisor"
         supervisor TRUE
         children []
       }

  6. このファイルを以下に配置する:
       <Webotsプロジェクト>/controllers/npc_trigger_supervisor/npc_trigger_supervisor.py

  7. HIDDEN_POSITION / CROSSING_START_POSITION / CROSSING_END_POSITION を
     実際に配置した座標に合わせて書き換える
     （webots_yolo_bridge_controller.py の Config.INTERSECTION_POSITION も
       CROSSING_START_POSITION に合わせておくと、IRLの
       distance_to_intersection_m と挙動が一致しやすい）
================================================================================
"""

import math

from controller import Supervisor


# ==============================================================================
# 0. CONFIG
# ==============================================================================
TIME_STEP = 32  # world の WorldInfo.basicTimeStep と合わせること

TRIGGER_DISTANCE_M = 15.0       # 自車がこの距離まで近づいたら飛び出し開始
DASH_DURATION_SEC = 1.2         # 飛び出しにかける時間（横断速度の目安: 距離/この値）
RESET_DELAY_SEC = 5.0           # 横断完了後、隠れ位置に戻すまでの待機時間

# --- 歩行者の隠れ位置(建物の陰)と、横断先(道路の反対側)の座標 [x, y, z] ---
# ワールド上の実際の建物・道路配置に合わせて必ず書き換えること
HIDDEN_POSITION = [8.0, 1.27, -14.0]
CROSSING_START_POSITION = [-59.3, 25.48, 1.3]
CROSSING_END_POSITION = [-59.3, -57.21, 1.3]


def distance_3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def lerp(p1, p2, t: float):
    t = max(0.0, min(1.0, t))
    return [a + (b - a) * t for a, b in zip(p1, p2)]


# ==============================================================================
# 1. 初期化
# ==============================================================================
supervisor = Supervisor()

ego_node = supervisor.getFromDef("EGO_VEHICLE")
pedestrian_node = supervisor.getFromDef("PEDESTRIAN_NPC")

if ego_node is None:
    raise RuntimeError(
        "DEF EGO_VEHICLE が見つかりません。Priusノードの手前に DEF EGO_VEHICLE を追加してください。"
    )
if pedestrian_node is None:
    raise RuntimeError(
        "DEF PEDESTRIAN_NPC が見つかりません。PedestrianノードにDEFを設定してください。"
    )

pedestrian_translation_field = pedestrian_node.getField("translation")
pedestrian_translation_field.setSFVec3f(HIDDEN_POSITION)

STATE_IDLE = "IDLE"
STATE_DASHING = "DASHING"
STATE_WAITING_RESET = "WAITING_RESET"

state = STATE_IDLE
dash_start_time = 0.0
reset_trigger_time = 0.0

print("[INFO] npc_trigger_supervisor 起動。自車接近を監視します。")
print(f"[INFO] トリガー距離: {TRIGGER_DISTANCE_M}m / 隠れ位置: {HIDDEN_POSITION}")


# ==============================================================================
# 2. メインループ
# ==============================================================================
while supervisor.step(TIME_STEP) != -1:
    sim_time = supervisor.getTime()
    ego_position = ego_node.getPosition()  # [x, y, z] ワールド座標

    distance_to_trigger = distance_3d(ego_position, CROSSING_START_POSITION)

    if state == STATE_IDLE:
        if distance_to_trigger <= TRIGGER_DISTANCE_M:
            state = STATE_DASHING
            dash_start_time = sim_time
            print(f"[EVENT t={sim_time:.2f}s] 自車が{distance_to_trigger:.1f}mまで接近 → 歩行者飛び出し開始")

    elif state == STATE_DASHING:
        elapsed = sim_time - dash_start_time
        t = elapsed / DASH_DURATION_SEC
        new_position = lerp(CROSSING_START_POSITION, CROSSING_END_POSITION, t)
        pedestrian_translation_field.setSFVec3f(new_position)

        if t >= 1.0:
            state = STATE_WAITING_RESET
            reset_trigger_time = sim_time
            print(f"[EVENT t={sim_time:.2f}s] 歩行者の横断完了")

    elif state == STATE_WAITING_RESET:
        if sim_time - reset_trigger_time >= RESET_DELAY_SEC:
            pedestrian_translation_field.setSFVec3f(HIDDEN_POSITION)
            state = STATE_IDLE
            print(f"[EVENT t={sim_time:.2f}s] 歩行者を隠れ位置へリセット。次の接近を待機します。")