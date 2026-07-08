import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import os

# デュエプレのウィンドウ名
WINDOW_TITLE = "デュエプレ"

# 読み込むテンプレート画像一覧（ファイル名と表示ラベル）
TEMPLATES = {
    "first.png": ("FIRST TURN (先攻)", (255, 255, 0)),   # 水色枠
    "second.png": ("SECOND TURN (後攻)", (255, 165, 0)), # オレンジ枠
    "win.png": ("WIN!! (勝利)", (0, 255, 0)),            # 緑枠
    "lose.png": ("LOSE... (敗北)", (0, 0, 255))          # 赤枠
}

def load_templates():
    """ フォルダ内のテンプレート画像を読み込む """
    loaded = {}
    for filename, (label, color) in TEMPLATES.items():
        if os.path.exists(filename):
            # 画像を読み込み（カラーで比較するためそのまま読み込む）
            img = cv2.imread(filename)
            loaded[filename] = {"img": img, "label": label, "color": color}
            print(f"✅ テンプレート読み込み成功: {filename} ({label})")
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
            
    return frame, detected_list

def start_capture():
    print("--- デュエプレ自動戦績記録 AIトラッカー ---")
    templates = load_templates()
    print("---------------------------------------------")
    
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        print(f"❌ 「{WINDOW_TITLE}」のウィンドウが見つかりません。")
        return

    target_window = windows[0]
    print(f"✅ ウィンドウを発見！監視とAI画像認識を開始します。")

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
            
            # --- AI画像認識を実行 ---
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