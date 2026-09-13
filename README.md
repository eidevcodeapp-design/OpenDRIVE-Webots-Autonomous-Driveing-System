# OpenDRIVE-Webots-Autonomous-Driveing-System
本プロジェクトは、GPU非搭載（CPU環境）の制約下において、実世界を模した高精度マップ（OpenDRIVE形式）をベースに、「知覚・計画・制御」の全レイヤーを自前で構築した自律走行システムの開発記録です。

---

## 📌 プロジェクトの概要

大手自動運転企業のアーキテクチャや安全モデルを参考に、限られた計算資源（CPUのみ）でリアルタイムかつ安全に動作する自律走行システムを実装しました。

* **HDマップ（OpenDRIVE / .xodr）の自動解析:** 座標のハードコーディングを行わず、XMLから道路網・ジャンクション・車線・信号情報を動的にパースしてグラフ構造を構築。
* **ダイクストラ法による経路探索 & GUI選択:** 2D鳥瞰図GUI上でスタートおよびゴール地点を直感的に選択可能。
* **IRL（逆強化学習）の応用:** 死角交差点等における人間らしい「減速・一時停止・安全確認」プロセスを、データ長の最適化により正確に再現。
* **YOLO ＋ RSS（責任敏感安全モデル）:** 視覚的な歩行者検知と数理的な安全マージン（反応時間・最小制動距離）に基づく絶対安全の担保。
* **GPS ＋ Pure Pursuit:** 幾何学的制御モデルによる滑らかな経路追従と、ゴール地点（ビル横など）への自動駐車・停止保持。

---

## 🛠 使用している主な技術

* **言語:** Python 3.8+
* **シミュレータ:** Webots (Driver API)
* **マップ・幾何学:** OpenDRIVE (`.xodr`), NumPy, SciPy (`cKDTree`, 3次スプライン補間 `splprep/splev`)
* **コンピュータビジョン & AI:** Ultralytics YOLOv8, ONNX Runtime, OpenCV
* **可視化・GUI:** Matplotlib (`matplotlib.widgets.Button`)
* **並行処理:** Python `threading` (YOLO非同期推論によるカクつき防止)

---

## 💻 必要な環境変数やコマンド一覧

### 📦 依存ライブラリのインストール
```bash
pip install numpy scipy opencv-python matplotlib ultralytics

---

### ⚙️ Webots 実行環境の設定
Webots に同梱の Python 環境を使う場合は追加設定不要ですが、システム側の Python から Driver API を使う場合は以下を設定してください。
```bash
export WEBOTS_HOME=/path/to/Webots
export PYTHONPATH=$WEBOTS_HOME/lib/controller/python:$PYTHONPATH
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller:$LD_LIBRARY_PATH
```

### 🤖 YOLOモデルの ONNX エクスポート（初回のみ）
CPU推論を高速化するため、PyTorchモデル（`yolov8n.pt`）を ONNX 形式（`yolov8n.onnx`）へ変換します。
```bash
yolo export model=yolov8n.pt format=onnx
```

### ▶️ コントローラの実行
Webots のロボット（車両）に本コントローラを割り当てた状態でシミュレーションを再生すると、`webots_integrated_controller.py` が自動的に実行されます。単体で構文・ロジックだけを確認したい場合は以下も利用できます。
```bash
python opendrive_parser.py   # パーサ＆グラフ構築の単体テスト
python route_planner.py      # 経路探索の単体テスト
```

---

## 📂 ディレクトリ構成

```
.
├── webots_integrated_controller.py  # メイン制御（FSM、YOLO非同期、Pure Pursuit、RSS統合）
├── opendrive_parser.py              # OpenDRIVE（.xodr）パーサ、グラフ構造化、サンプル都市生成
├── route_planner.py                 # ダイクストラ法による経路探索、ウェイポイント生成、トリガー管理
├── map_gui.py                       # 2D鳥瞰図GUI（スタート/ゴール選択、cKDTree高速ノード検索）
├── world_top_view.png               # 背景用の街並み鳥瞰図画像
├── yolov8n.pt                       # YOLOv8標準モデル重み
└── yolov8n.onnx                     # 高速化のためにエクスポートされたONNXモデル
```

---

## 🚀 セットアップ手順

* **Webots のインストール:** Webots をインストールし、Python から Webots の Driver API を利用できる環境を構築してください。必要に応じて Webots の実行ファイルへのパスを環境変数に追加するか、Webots 付属の Python 環境を使用してください。
* **リポジトリの配置:** `webots_integrated_controller.py` / `opendrive_parser.py` / `route_planner.py` / `map_gui.py` / `world_top_view.png` を同一のプロジェクトディレクトリに配置してください。
* **YOLOモデルの準備:** 初回実行時には、Ultralytics によって YOLOv8n のモデル重み（`yolov8n.pt`）が必要に応じてダウンロードされます。CPU環境での推論負荷を軽減するため、上記コマンドで ONNX 形式（`yolov8n.onnx`）へエクスポートして使用します。

---

## ⚠️ 開発で直面した課題と解決策

開発過程では、主に以下の3つの問題に直面しました。

### 🧠 1. IRL / 交差点減速の学習・再現における問題

* **問題:** IRL による走行挙動の学習を行った際、交差点手前で減速せず、そのまま交差点を通過してしまう問題が発生。交差点通過後の加速シーンばかりがモデルに再現され、「減速・一時停止・安全確認」という重要な挙動が十分に学習されなかった。
* **原因:** トレーニングデータのシーケンス長が長すぎたこと。走行全体のデータをそのまま学習に使用すると、交差点手前の短い減速区間がデータ全体の中で相対的に小さくなり、重要な減速トリガーがノイズとして埋もれてしまう。
* **解決策:** 単純にトレーニングデータの総量を削減するのではなく、**「学習させたい場面」をピンポイントで切り出す**方法を採用。交差点への接近／減速開始／減速／一時停止／安全確認、といった重要な区間を重点的に抽出し、データのシーケンス長を最適化した。その結果、交差点手前の減速・停止という意図した挙動をモデルが正確に捉えられるようになった。

### 👁️ 2. YOLO単体での車線追従による逸脱問題

* **問題:** 当初は YOLO による白線検出・周辺物体認識を利用して車両の走行経路を決定する構成を検討していたが、道路上の影・ガードレール・道路境界・白線の誤検出・一時的な物体検出の失敗などにより認識結果が不安定になり、車両が規定のコースから大きく逸脱する問題が発生した。
* **解決策（アーキテクチャの変更）:** YOLOの役割を車線追従から危険察知へ完全に分離。YOLOは歩行者・周辺物体の認識専用とし、検出結果はRSSによる安全判定へ利用する。経路追従はYOLOから切り離し、OpenDRIVEから自動抽出したWaypointsを使って **GPS ＋ Pure Pursuit** によって車両を経路上に追従させる構成へ変更した。

制御アーキテクチャ：
```
OpenDRIVE
    ↓
道路ネットワーク生成
    ↓
ダイクストラ法
    ↓
経路探索
    ↓
Waypoints生成
    ↓
GPSによる自己位置取得
    ↓
Pure Pursuit
    ↓
ステアリング制御
```

最終的な役割分担：

| モジュール | 役割 |
|---|---|
| OpenDRIVE | 道路ネットワーク構築 |
| Dijkstra | 経路探索 |
| GPS | 自己位置取得 |
| Pure Pursuit | 経路追従 |
| YOLO | 歩行者・周辺物体認識 |
| RSS | 危険判定・緊急停止 |
| FSM | 走行状態管理 |
| IRL | 人間らしい減速・停止挙動 |

この構成にすることで、認識・経路計画・車両制御・安全判断の責務を分離し、YOLOの誤認識によって経路追従そのものが不安定になる問題を抑制した。

### 💻 3. CPU環境におけるYOLO推論の負荷とカクつき

* **問題:** GPUを搭載していないCPU環境で、メインのWebots制御ループ内から毎フレームYOLO推論を実行していたため、CPU使用率の上昇・Webotsの処理速度低下・シミュレーションのカクつき・車両制御の遅延が発生した。
* **解決策①（非同期化）:** YOLOの推論処理をメインの制御ループから分離し、Pythonの `threading` を利用して別スレッドで実行する構成へ変更。さらに毎フレーム推論するのではなく、数フレームに1回程度の頻度で推論することでCPU負荷を削減した。
* **解決策②（ONNX化）:** YOLOv8のPyTorchモデルをONNX形式へエクスポートし、ONNX Runtimeを利用して推論することで、CPU環境における推論処理を軽量化した。

非同期推論の構成：
```
Webots Main Control Loop
        │
        ├── GPS
        ├── Pure Pursuit
        ├── FSM
        └── Vehicle Control
                │
                │
        ┌───────▼────────┐
        │ YOLO Thread    │
        │ 非同期推論     │
        └───────┬────────┘
                ↓
        Object Detection
                ↓
        RSS Safety Judgment
```

これにより、YOLOの推論処理が車両制御ループを直接停止させることを防いだ。

---

## 🎯 最終的なシステム構成

```
                    ┌─────────────────────┐
                    │       Webots        │
                    │    自動運転環境      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Main Control Loop │
                    │         FSM         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ↓                ↓                ↓
          OpenDRIVE           GPS             YOLO
              ↓                ↓                ↓
          Dijkstra       Pure Pursuit       Object
              ↓                ↓            Detection
          Waypoints            │                ↓
              │                │               RSS
              └────────────────┘                ↓
                       │                    Safety Control
                       ↓
                 Vehicle Control
```

---

## 🏆 技術的なポイント

* OpenDRIVEから道路ネットワークを自動構築
* ダイクストラ法による経路探索
* GPS ＋ Pure Pursuitによる安定した経路追従
* IRLによる交差点手前の減速・停止挙動
* YOLOによる歩行者・周辺物体認識
* RSSによる安全判定
* FSMによるシステム全体の状態管理
* YOLOの非同期推論によるCPU負荷の削減
* ONNX Runtimeによる推論の高速化

これらを組み合わせることで、限られたCPU計算資源でもリアルタイム性と安全性を両立する自律走行システムを構築しました。
