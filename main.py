import json
import os
import sys
import threading
import time

import cv2
import numpy as np
from pyray import *

LEBAR_SC = 1012      # tinggi layar
PANJANG_SC = 1600    # lebar layar
BOX_COUNT = 4
GAPBX = 300
SCROLL_SPEED = 4.0   # mouse wheel sisaan dasprog
BIRD_MAX_Y = LEBAR_SC - 150
HITBOX_SHRINK = 40

HSV_FILE = "hsv.json"

# DEFAULT_HSV = {"lower": [125, 25, 30], "upper": [170, 160, 200]}
# DEFAULT_HSV = {"lower": [5, 150, 100], "upper": [25, 255, 255]}
DEFAULT_HSV = {"lower": [172, 100, 100], "upper": [12, 220, 230]}
DEBUG_SCALE = 1.0    # 1.0 = 640x240, 0.75 = 480x180, 0.5 = 320x120

def in_range_hsv(hsv, lower, upper):

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    a = cv2.inRange(hsv, lower, np.array([179, upper[1], upper[2]]))
    b = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]), upper)
    return cv2.bitwise_or(a, b)


def load_hsv():
    if os.path.exists(HSV_FILE):
        with open(HSV_FILE) as f:
            d = json.load(f)
        return np.array(d["lower"]), np.array(d["upper"])
    return np.array(DEFAULT_HSV["lower"]), np.array(DEFAULT_HSV["upper"])


class HSVTracker(threading.Thread):
    MIN_AREA = 150  # luas minimum kontur (frame 320x240)
    KERNEL = np.ones((5, 5), np.uint8)

    def __init__(self, cam_index=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.lower, self.upper = load_hsv()
        self.y_norm = None       # posisi Y objek, 0..1 (None = tidak terdeteksi)
        self.debug_frame = None
        self.debug = True        # jendela kamera debug aktif sejak awal (tombol D = toggle)
        self.running = True
        self.available = self.cap.isOpened()

    def run(self):
        while self.running and self.available:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = in_range_hsv(hsv, self.lower, self.upper)
            mask = cv2.erode(mask, None, iterations=1)
            # closing: menutup lubang kecil (titik putih sarung tangan) agar jadi satu objek
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.KERNEL, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            box = None  # (x, y, w, h) kotak di sekeliling area warna terbesar
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > self.MIN_AREA:
                    box = cv2.boundingRect(c)

            h_frame = frame.shape[0]
            if box:
                x, y, w, h = box
                cx, cy = x + w // 2, y + h // 2   # nilai tengah kotak
                self.y_norm = cy / h_frame
            else:
                self.y_norm = None

            if self.debug:
                view = frame.copy()
                # garis zona kontrol (15%..85% tinggi frame)
                for ratio in (0.15, 0.85):
                    yy = int(h_frame * ratio)
                    cv2.line(view, (0, yy), (view.shape[1], yy), (255, 128, 0), 1)
                if box:
                    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.drawMarker(view, (cx, cy), (0, 0, 255),
                                   cv2.MARKER_CROSS, 16, 2)
                    cv2.putText(view, f"center=({cx},{cy}) y={self.y_norm:.2f}",
                                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(view, "TIDAK TERDETEKSI", (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 255), 1, cv2.LINE_AA)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combo = np.hstack([view, mask_bgr])
                self.debug_frame = cv2.resize(combo, None, fx=2, fy=2,
                                              interpolation=cv2.INTER_NEAREST)

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def calibrate():
    """Alat bantu: atur slider sampai hanya objekmu yang putih di mask, tekan S untuk simpan."""
    lower, upper = load_hsv()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Kalibrasi HSV")
    names = [("H low", lower[0], 179), ("H high", upper[0], 179),
             ("S low", lower[1], 255), ("S high", upper[1], 255),
             ("V low", lower[2], 255), ("V high", upper[2], 255)]
    for n, v, m in names:
        cv2.createTrackbar(n, "Kalibrasi HSV", int(v), m, lambda x: None)

    print("S = simpan ke hsv.json, Q = keluar")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        g = lambda n: cv2.getTrackbarPos(n, "Kalibrasi HSV")
        lo = np.array([g("H low"), g("S low"), g("V low")])
        hi = np.array([g("H high"), g("S high"), g("V high")])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = in_range_hsv(hsv, lo, hi)
        cv2.imshow("Kalibrasi HSV",
                   np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            with open(HSV_FILE, "w") as f:
                json.dump({"lower": lo.tolist(), "upper": hi.tolist()}, f)
            print("Tersimpan:", lo.tolist(), hi.tolist())
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# kode game 
class Box:
    def __init__(self):
        self.x = 0.0
        self.abv_y = 0.0
        self.blw_y = 0.0
        self.abv_w = self.abv_h = 0
        self.blw_w = self.blw_h = 0

    def abv_rect(self):
        return Rectangle(self.x, self.abv_y, self.abv_w, self.abv_h)

    def blw_rect(self):
        return Rectangle(self.x, self.blw_y, self.blw_w, self.blw_h)

    def randomize_y(self):
        self.blw_y = float(get_random_value(300, LEBAR_SC - 100))
        self.abv_y = self.blw_y - self.blw_h - 200


def init_boxes(abvbox, blwbox):
    """Padanan boxmovement() di C."""
    boxes = []
    for i in range(BOX_COUNT):
        b = Box()
        b.abv_w, b.abv_h = abvbox.width, abvbox.height
        b.blw_w, b.blw_h = blwbox.width, blwbox.height
        b.x = PANJANG_SC + i * (abvbox.width + GAPBX)
        b.randomize_y()
        boxes.append(b)
    return boxes


class Game:
    def __init__(self, tracker):
        self.tracker = tracker
        self.game_over = False
        self.score = 0
        self.scored = [False] * BOX_COUNT
        self.txt_welcome_pos = PANJANG_SC // 2 - 150

        # visuals()
        self.bg = load_texture("pictures/background.png")
        self.bird = load_texture("pictures/bird.png")
        self.abvbox = load_texture("pictures/abv boxes scrolly.png")
        self.blwbox = load_texture("pictures/blw boxes scrolly.png")

        self.bird_w = self.bird.width // 2   # sprite sheet 2 frame
        self.bird_h = self.bird.height
        self.pos_x = 350.0
        self.pos_y = float(LEBAR_SC // 2 - self.bird_h // 2)
        self.target_y = self.pos_y

        self.current_frame = 0
        self.frames_counter = 0
        self.frame_rec = Rectangle(0, 0, self.bird_w, self.bird_h)

        self.boxes = init_boxes(self.abvbox, self.blwbox)

    # input
    def update_bird_position(self):
        # mouse wheel
        wheel = get_mouse_wheel_move()
        if wheel != 0:
            self.target_y -= int(wheel * SCROLL_SPEED)

        # Kontrol utama: posisi Y objek berwarna dari kamera
        ny = self.tracker.y_norm if self.tracker else None
        if ny is not None:
            # pakai rentang tengah kamera (15%..85%) supaya tidak perlu sampai tepi frame
            t = min(max((ny - 0.15) / 0.70, 0.0), 1.0)
            self.target_y = t * BIRD_MAX_Y

        self.target_y = min(max(self.target_y, 0), BIRD_MAX_Y)
        # smoothing agar gerakan tidak patah-patah
        self.pos_y += (self.target_y - self.pos_y) * 0.3

    # bird
    def bird_animation(self):
        frames_speed = 8
        self.frames_counter += 1
        if self.frames_counter >= 60 // frames_speed:
            self.frames_counter = 0
            self.current_frame = (self.current_frame + 1) % 2
            self.frame_rec.x = float(self.current_frame * self.bird_w)

    # obstavcle 
    def box_pos_update(self):
        for i, b in enumerate(self.boxes):
            b.x -= 4
            if b.x < -b.abv_w:
                b.randomize_y()
                b.x = float(PANJANG_SC)
                self.scored[i] = False

    # game system
    def game_system(self):
        s = HITBOX_SHRINK
        bird_rect = Rectangle(self.pos_x + s, self.pos_y + s, self.bird_w - s*2, self.bird_h - s*2)
        if not self.game_over:
            for i, b in enumerate(self.boxes):
                if (check_collision_recs(bird_rect, b.abv_rect())
                        or check_collision_recs(bird_rect, b.blw_rect())):
                    self.game_over = True
                    break
                elif int(b.x) < int(self.pos_x) and not self.scored[i]:
                    self.scored[i] = True
                    self.score += 1
        else:
            if is_key_pressed(KEY_ENTER):
                self.reset_game()

    def reset_game(self):
        self.game_over = False
        self.score = 0
        self.scored = [False] * BOX_COUNT
        for i, b in enumerate(self.boxes):
            b.x = PANJANG_SC + i * (self.abvbox.width + GAPBX)
            b.randomize_y()

    # draw
    def draw_game(self):
        draw_texture(self.bg, 0, 0, WHITE)
        if not self.game_over:
            draw_texture_rec(self.bird, self.frame_rec,
                             Vector2(self.pos_x, self.pos_y), WHITE)
            draw_rectangle_lines(int(self.pos_x + HITBOX_SHRINK), int(self.pos_y + HITBOX_SHRINK),
                     self.bird_w - 2 * HITBOX_SHRINK, self.bird_h - 2 * HITBOX_SHRINK, RED)
            self.txt_welcome_pos -= 4
            draw_text("WELCOME TO SCROLLY BIRD", self.txt_welcome_pos,
                      LEBAR_SC // 2 - 25, 55, WHITE)
            for b in self.boxes:
                draw_texture(self.abvbox, int(b.x), int(b.abv_y), WHITE)
                draw_texture(self.blwbox, int(b.x), int(b.blw_y), WHITE)
            draw_text(f"SCORE = {self.score}", 20, 20, 50, WHITE)
        else:
            def centered(text, y):
                draw_text(text, PANJANG_SC // 2 - measure_text(text, 40) // 2,
                          y, 40, WHITE)
            centered("Game Over", LEBAR_SC // 2 - 70)
            centered(f"Score = {self.score}", LEBAR_SC // 2 - 30)
            centered("Press Enter to Retry", LEBAR_SC // 2 + 10)

        # indikator status kamera
        if self.tracker and self.tracker.available:
            ok = self.tracker.y_norm is not None
            draw_circle(PANJANG_SC - 30, 30, 10, GREEN if ok else RED)

    def unload(self):
        for t in (self.bg, self.bird, self.abvbox, self.blwbox):
            unload_texture(t)


# main
def main():
    tracker = HSVTracker()
    if tracker.available:
        tracker.start()
    else:
        print("Kamera tidak ditemukan, pakai mouse wheel saja.")

    init_window(PANJANG_SC, LEBAR_SC, "Scrolly Bird")
    set_target_fps(60)
    game = Game(tracker)

    debug_open = False
    while not window_should_close():
        if not game.game_over:
            game.update_bird_position()
            game.bird_animation()
            game.box_pos_update()
        game.game_system()

        begin_drawing()
        clear_background(WHITE)
        game.draw_game()
        end_drawing()

        # debug window
        if is_key_pressed(KEY_D):
            tracker.debug = not tracker.debug
            if not tracker.debug and debug_open:
                cv2.destroyWindow("Debug HSV")
                debug_open = False
        if tracker.debug and tracker.debug_frame is not None:
            cv2.imshow("Debug HSV", tracker.debug_frame)
            cv2.waitKey(1)
            debug_open = True

    tracker.stop()
    cv2.destroyAllWindows()
    game.unload()
    close_window()


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        main()