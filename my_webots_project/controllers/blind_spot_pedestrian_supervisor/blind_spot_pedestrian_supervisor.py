"""
================================================================================
 blind_spot_pedestrian_supervisor.py
 死角交差点での歩行者飛び出しシナリオ用 Supervisorコントローラ

【役割】
  1. シミュレーション開始/リセット(Revert)のたびに、歩行者(Pedestrian)を
     死角の初期位置(建物の裏)へ強制的に戻す。
     → Webotsはワールドをリセットしてもキネマティックに動かしたノードの
       translation/rotationは「前回動かした場所」に残ってしまう。
       このコントローラ自体は世界がリセットされるたびに再起動されるので、
       スクリプト冒頭でリセット処理を行えば毎回必ず初期位置に戻る。
  2. 自車(Ego Vehicle)が死角交差点の手前 TRIGGER_DISTANCE_M 以内に
     入ったことを毎ステップ監視する。
  3. 条件を満たした瞬間に、歩行者を一定速度で道路を横切る方向へ
     直線的に移動させる(飛び出しイベント)。
  4. (おまけ) キーボードの 'R' キーで、ワールドをRevertしなくても
     その場で歩行者リセット+シナリオ再アームができる。
     何度もテイクを取り直すテスト運用を想定した補助機能。

【配置方法】
  <Webotsプロジェクト>/controllers/blind_spot_pedestrian_supervisor/blind_spot_pedestrian_supervisor.py

【事前準備】
  - このスクリプトを controller に設定したRobotノード(supervisorフィールド=TRUE)を
    ワールドに追加しておくこと(詳細はチャット本文のScene Tree手順を参照)。
  - 自車ノードに DEF EGO_VEHICLE、歩行者ノードに DEF PEDESTRIAN_NPC を
    付けておくこと(下のConfigの名前と一致させること)。
  - 歩行者ノードに歩行アニメーション用の独自controllerが設定されている場合は
    'void' にしておくこと(このスクリプトが直接translationを操作するため)。
================================================================================
"""

import math

from controller import Supervisor


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class Config:
    # --- ノードのDEF名(Scene Treeで設定したものと一致させること) ---
    EGO_DEF_NAME = "EGO_VEHICLE"
    PEDESTRIAN_DEF_NAME = "PEDESTRIAN_NPC"

    # --- 死角交差点(webots_yolo_bridge_controller.py と同じ値を使うと整合が取れる) ---
    INTERSECTION_POSITION_XZ = [-74.23, 36.4]
    # 歩行者が飛び出すタイミング
    TRIGGER_DISTANCE_M = 4.5

    # --- 歩行者の初期位置(死角/建物の裏)。Scene Treeで手動配置して
    #     controlパネルに表示された値をそのままコピーすること。 ---
    PEDESTRIAN_INITIAL_TRANSLATION = [-59.42, 26.23, 1.3]   # [x, y, z] 例。必ず実際の値に置き換える
    PEDESTRIAN_INITIAL_ROTATION = [0.0, 0.0, 1.0, 1.57]  # [x, y, z, angle(rad)] 例。必ず実際の値に置き換える

    # --- 飛び出し(横断)の動き ---
    # 自車の進行方向(道路)に対して「横切る」方向の単位ベクトル [X成分, Z成分]。
    # 例: 道路がX軸方向に伸びている場合、横断方向は概ねZ軸方向になるので [0.0, 1.0] 等。
    # 実際の道路の向きに合わせて必ず調整すること。
    PEDESTRIAN_WALK_DIRECTION_XY = [0.0, 1.0]  # 自車は、歩行者に対して、x軸方向に横切っていた
    PEDESTRIAN_WALK_SPEED_MPS = 3.5       # 1.5~2.5 m/s の範囲で調整可
    PEDESTRIAN_WALK_DISTANCE_M = 31.0      # この距離だけ歩いたら停止(道路の幅目安)

    # --- Rキーでの手動リアーム機能(任意) ---
    ENABLE_MANUAL_REARM_KEY = True
    REARM_KEY_CHAR = "R"


# ==============================================================================
# 1. 初期化
# ==============================================================================
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

ego_node = supervisor.getFromDef(Config.EGO_DEF_NAME)
if ego_node is None:
    raise RuntimeError(
        f"DEF '{Config.EGO_DEF_NAME}' の自車ノードが見つかりません。"
        f"Scene Treeで自車ノードにこのDEF名を付けてください。"
    )

pedestrian_node = supervisor.getFromDef(Config.PEDESTRIAN_DEF_NAME)
if pedestrian_node is None:
    raise RuntimeError(
        f"DEF '{Config.PEDESTRIAN_DEF_NAME}' の歩行者ノードが見つかりません。"
        f"Scene Treeで歩行者ノードにこのDEF名を付けてください。"
    )

ego_translation_field = ego_node.getField("translation")
pedestrian_translation_field = pedestrian_node.getField("translation")
pedestrian_rotation_field = pedestrian_node.getField("rotation")

if Config.ENABLE_MANUAL_REARM_KEY:
    supervisor.keyboard.enable(timestep)


# ==============================================================================
# 2. ユーティリティ
# ==============================================================================
def distance_2d_xz(p1_xz, p2_xz) -> float:
    return math.hypot(p1_xz[0] - p2_xz[0], p1_xz[1] - p2_xz[1])


def reset_pedestrian_to_initial_position():
    """
    歩行者を死角の初期位置へ強制的に戻す。
    シミュレーション開始時・Revert時に必ず呼ばれるほか、
    手動リアームキーが押された時にも呼ばれる。
    """
    pedestrian_translation_field.setSFVec3f(Config.PEDESTRIAN_INITIAL_TRANSLATION)
    pedestrian_rotation_field.setSFRotation(Config.PEDESTRIAN_INITIAL_ROTATION)
    # 物理エンジンに蓄積された速度・角速度もリセットしておく(念のため)
    pedestrian_node.resetPhysics()


# ==============================================================================
# 3. シミュレーション開始時の強制リセット
#    (Webotsはワールドをリセットしても「前回動かした場所」がそのまま残るため、
#     このコントローラが再起動されるたびに必ずここを通ることで解決する)
# ==============================================================================
reset_pedestrian_to_initial_position()
print("[Supervisor] 歩行者を死角の初期位置へリセットしました。")

triggered = False
walked_distance_m = 0.0


# ==============================================================================
# 4. メインループ
# ==============================================================================
print("[Supervisor] 監視を開始します。自車が死角交差点に接近すると歩行者が飛び出します。")

while supervisor.step(timestep) != -1:

    # --- 任意: Rキーでいつでもその場でリセット+再アーム(Revert不要) ---
    if Config.ENABLE_MANUAL_REARM_KEY:
        key = supervisor.keyboard.getKey()
        if key == ord(Config.REARM_KEY_CHAR):
            reset_pedestrian_to_initial_position()
            triggered = False
            walked_distance_m = 0.0
            print("[Supervisor] 手動リアーム: 歩行者を初期位置へ戻し、シナリオを再度待機状態にしました。")

    # --- 自車位置の取得と交差点までの距離判定 ---
    ego_pos = ego_translation_field.getSFVec3f()
    ego_pos_xz = [ego_pos[0], ego_pos[1]]        # 修正：yを奥行として取得する
    distance_to_intersection_m = distance_2d_xz(ego_pos_xz, Config.INTERSECTION_POSITION_XZ)

    if not triggered and distance_to_intersection_m <= Config.TRIGGER_DISTANCE_M:
        triggered = True
        print(
            f"[Supervisor] 自車が交差点から{distance_to_intersection_m:.1f}mに接近。"
            f"歩行者の飛び出しを開始します。"
        )

    # --- 飛び出し中: 一定速度で横断方向へ直線的に移動 ---
    if triggered and walked_distance_m < Config.PEDESTRIAN_WALK_DISTANCE_M:
        dt_sec = timestep / 1000.0
        step_dist_m = Config.PEDESTRIAN_WALK_SPEED_MPS * dt_sec

        current_pos = pedestrian_translation_field.getSFVec3f()
        # new_pos の更新部分を以下に修正
        new_pos = [
            current_pos[0] + Config.PEDESTRIAN_WALK_DIRECTION_XY[0] * step_dist_m,  # X移動
            current_pos[1] + Config.PEDESTRIAN_WALK_DIRECTION_XY[1] * step_dist_m,  # Y移動（奥行き）
            current_pos[2],  # Z軸（高さ）を保持して空中浮遊を防ぐ
        ]
        pedestrian_translation_field.setSFVec3f(new_pos)
        walked_distance_m += step_dist_m

        if walked_distance_m >= Config.PEDESTRIAN_WALK_DISTANCE_M:
            print(f"[Supervisor] 歩行者が横断完了({walked_distance_m:.1f}m移動)。停止します。")