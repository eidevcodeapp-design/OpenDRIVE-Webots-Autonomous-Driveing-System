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

