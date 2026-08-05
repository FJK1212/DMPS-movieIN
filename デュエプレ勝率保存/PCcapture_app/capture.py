import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import os
import json
import sys
import threading

from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# デュエプレのウィンドウ名
WINDOW_TITLE = "デュエプレ"

# ローカルサーバー設定（mein.html への配信 + 検出結果APIを兼ねる）
SERVER_PORT = 8765

if getattr(sys, "frozen", False):
    # exe化した場合、__file__ではなくexe自身の場所を基準にする
    # （PCcapture_appフォルダ内にexeを置く運用のため、その親フォルダがルート）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 読み込むテンプレート画像一覧（ファイル名・表示ラベル・種別(turn/result)・値）
TEMPLATES = {
    "first.png":  {"label": "FIRST TURN (先攻)",  "color": (255, 255, 0), "kind": "turn",   "value": "先攻"},
    "second.png": {"label": "SECOND TURN (後攻)", "color": (255, 165, 0), "kind": "turn",   "value": "後攻"},
    "win.png":    {"label": "WIN!! (勝利)",       "color": (0, 255, 0),   "kind": "result", "value": "勝ち"},
    "lose.png":   {"label": "LOSE... (敗北)",     "color": (0, 0, 255),   "kind": "result", "value": "負け"},
}

# 検出結果の共有ステート（capture側のスレッドとHTTPサーバー側のスレッドで共有）
state_lock = threading.Lock()
status = {"armed": True, "turn": None, "result": None}


def update_status(kind, value):
    """ 検出結果をステートに反映する。armedがFalseの間（結果確定後、mein.html側がackするまで）は無視する """
    with state_lock:
        if not status["armed"]:
            return
        if status[kind] is None:
            status[kind] = value
            if kind == "result":
                status["armed"] = False  # 結果確定。次はmein.html側のackを待つ


class StatusHandler(SimpleHTTPRequestHandler):
    """ mein.html等の静的配信 + /api/status, /api/ack を提供するハンドラ """

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            with state_lock:
                self._send_json(dict(status))
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/ack":
            with state_lock:
                status["armed"] = True
                status["turn"] = None
                status["result"] = None
            self._send_json({"ok": True})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # コンソールを静かにする


def start_server():
    handler = partial(StatusHandler, directory=PROJECT_ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", SERVER_PORT), handler)
    print(f"✅ ローカルサーバー起動: http://127.0.0.1:{SERVER_PORT}/DMPS/mein.html （このURLをブラウザで開いてください）")
    httpd.serve_forever()


def imread_unicode(path):
    """ cv2.imreadは日本語パスを読めないため、np.fromfile経由で読み込む """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def load_templates():
    """ フォルダ内のテンプレート画像を読み込む """
    loaded = {}
    for filename, meta in TEMPLATES.items():
        path = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(path):
            # 画像を読み込み（カラーで比較するためそのまま読み込む）
            img = imread_unicode(path)
            if img is None:
                print(f"⚠️ 画像の読み込みに失敗しました: {filename}")
                continue
            loaded[filename] = {**meta, "img": img}
            print(f"✅ テンプレート読み込み成功: {filename} ({meta['label']})")
        else:
            print(f"⚠️ 画像が見つかりません: {filename} (後から追加してもOKです)")
    return loaded

def match_and_draw(frame, templates, threshold=0.75):
    """ 画面内からテンプレートを探し、見つかれば枠を描く """
    detected_list = []
    
    for name, data in templates.items():
        tmpl = data["img"]
        if tmpl is None:
            continue
            
        # テンプレートマッチング実行
        res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 一致度がしきい値（0.75 = 75%以上）を超えていたら「見つけた！」と判定
        if max_val >= threshold:
            h, w = tmpl.shape[:2]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # 画面上にカラー枠線を描く
            cv2.rectangle(frame, top_left, bottom_right, data["color"], 3)
            # 枠線の上にラベル文字を描く
            cv2.putText(frame, data["label"], (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, data["color"], 2)

            detected_list.append(data["label"])
            update_status(data["kind"], data["value"])
            
    return frame, detected_list

def start_capture():
    print("--- デュエプレ自動戦績記録トラッカー ---")
    templates = load_templates()
    threading.Thread(target=start_server, daemon=True).start()
    print("---------------------------------------------")

    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        print(f"❌ 「{WINDOW_TITLE}」のウィンドウが見つかりません。")
        return

    target_window = windows[0]
    print(f"✅ ウィンドウを発見！監視とテンプレートマッチングを開始します。")

    # 連続で文字を出力しないための記憶用変数
    last_detected = []

    with mss.MSS() as sct:
        while True:
            if target_window.isMinimized:
                time.sleep(1)
                continue

            monitor = {
                "top": target_window.top,
                "left": target_window.left,
                "width": target_window.width,
                "height": target_window.height
            }

            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # --- テンプレートマッチングを実行 ---
            frame, detected = match_and_draw(frame, templates)

            # 新しい要素を検出したときだけコンソールに表示
            if detected and detected != last_detected:
                for d in detected:
                    print(f"🎉 検出しました: 【 {d} 】")
                last_detected = detected
            elif not detected:
                last_detected = []
            # ------------------------

            # 半分のサイズにしてウィンドウ表示
            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Duel Masters Tracker - Camera", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("監視を終了します。")
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_capture()