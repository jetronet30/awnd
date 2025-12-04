#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAGON TRACKER — INDUSTRIAL FINAL v15.4
START/STOP + სრულიად სტაბილური + აღარანაირი SyntaxError/UnboundLocalError
100% PyInstaller თავსებადი | 2025
"""

import sys
import os
import cv2
import time
import signal
import json
import logging
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import queue
import socket
from threading import Thread, Lock
from pathlib import Path


# ==================== PyInstaller ====================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ==================== LOGGING ====================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"wagon_tracker_{datetime.now():%Y%m%d}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("WAGON_TRACKER")


# ==================== CONFIG ====================
MODEL_PATH = resource_path("best.pt")
UNIQUE_WAGON_JSON = "unique_wagons.json"
TCP_SERVER_IP = "127.0.0.1"
TCP_SERVER_PORT = 45002
RECONNECT_DELAY = 10
OCR_COOLDOWN = 4.0
DETECTION_EVERY_N_FRAME = 4
ENABLE_GUI = True
HEADLESS = False

CAMERAS = [
    {"name": "cam 1", "url": "rtsp://admin:admin@192.168.1.11:554", "roi": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}},
    {"name": "cam 2", "url": "rtsp://admin:admin@192.168.1.11:554", "roi": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}},
]


# ==================== GLOBAL FLAGS ====================
running = True
detection_enabled = False
detection_lock = Lock()
cameras = []
tcp_socket = None


# ==================== OCR SETUP ====================
log.info("TrOCR მოდელის ჩატვირთვა...")
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
ocr_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
ocr_model.to(device)
torch.set_num_threads(8)
log.info(f"TrOCR მზადაა: {device}")

ocr_queue = queue.Queue(maxsize=6)


class Camera:
    def __init__(self, cfg, idx):
        self.cfg = cfg
        self.idx = idx + 1
        self.name = cfg["name"]

        self.latest_frame = None
        self.last_boxes = {}
        self.last_seen = {}
        self.wagon_numbers = {}
        self.unique_numbers = set()
        self.next_display_id = 1
        self.track_to_display = {}
        self.wagons_list = []

        self.frame_lock = Lock()
        self.data_lock = Lock()

        log.info(f"[{self.name}] YOLO მოდელის ჩატვირთვა...")
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()

        Thread(target=self.run, daemon=True, name=f"CAM-{self.name}").start()

    def run(self):
        cap = None
        fc = 0
        last_ocr = 0

        while running:
            try:
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(self.cfg["url"], cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    log.info(f"[{self.name}] RTSP დაკავშირებული")

                ret, frame = cap.read()
                if not ret:
                    log.warning(f"[{self.name}] ფრეიმი ვერ მიიღო")
                    cap.release()
                    cap = None
                    time.sleep(3)
                    continue

                with self.frame_lock:
                    self.latest_frame = frame.copy()

                fc += 1

                with detection_lock:
                    currently_enabled = detection_enabled

                if not currently_enabled:
                    time.sleep(0.01)
                    continue

                if fc % DETECTION_EVERY_N_FRAME != 0:
                    time.sleep(0.001)
                    continue

                h, w = frame.shape[:2]
                r = self.cfg["roi"]
                x1 = max(0, int(w * r["x1"]))
                y1 = max(0, int(h * r["y1"]))
                x2 = min(w, int(w * r["x2"]))
                y2 = min(h, int(h * r["y2"]))
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]

                results = self.model.track(
                    roi, persist=True, conf=0.3, tracker="botsort.yaml",
                    imgsz=640, verbose=False
                )[0]

                best_tid = None
                best_conf = 0.0

                with self.data_lock:
                    current = set()
                    for box in results.boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id.item())
                        current.add(tid)
                        conf = float(box.conf.item()) if hasattr(box.conf, 'item') else 0.0
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        gx1, gy1 = x1 + bx1, y1 + by1
                        gx2, gy2 = x1 + bx2, y1 + by2

                        self.last_boxes[tid] = (gx1, gy1, gx2, gy2)
                        self.last_seen[tid] = fc

                        if conf > best_conf:
                            best_conf = conf
                            best_tid = tid

                    for t in list(self.last_boxes.keys()):
                        if t not in current:
                            self.last_boxes.pop(t, None)
                            self.last_seen.pop(t, None)

                if (best_tid and best_conf > 0.75 and
                    best_tid not in self.wagon_numbers and
                    time.time() - last_ocr > OCR_COOLDOWN):

                    with detection_lock:
                        if not detection_enabled:
                            continue

                    last_ocr = time.time()
                    x1, y1, x2, y2 = self.last_boxes[best_tid]
                    if x2 - x1 > 120 and y2 - y1 > 45:
                        crop = frame[y1:y2, x1:x2]
                        try:
                            ocr_queue.put_nowait((crop, best_tid, self))
                        except queue.Full:
                            pass

                time.sleep(0.001)

            except Exception as e:
                log.error(f"[{self.name}] Run შეცდომა: {e}")
                time.sleep(0.1)


def ocr_worker():
    log.info("OCR Worker გაშვებული")
    while running:
        try:
            crop, tid, cam = ocr_queue.get(timeout=3)
        except queue.Empty:
            continue
        try:
            with torch.inference_mode():
                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 56))
                pixel_values = ocr_processor(img, return_tensors="pt").pixel_values.to(device)
                ids = ocr_model.generate(pixel_values, max_length=12)
                text = ocr_processor.batch_decode(ids, skip_special_tokens=True)[0]
                num = "".join(c for c in text if c.isdigit())

                if len(num) == 8 and num not in cam.unique_numbers:
                    with cam.data_lock:
                        cam.unique_numbers.add(num)
                        cam.wagon_numbers[tid] = num
                        if tid not in cam.track_to_display:
                            cam.track_to_display[tid] = cam.next_display_id
                            cam.next_display_id += 1
                        display_id = cam.track_to_display[tid]
                        cam.wagons_list.append({"id": display_id, "number": num})
                    log.info(f"[{cam.name}] ID{display_id} → {num}")
        except Exception as e:
            log.error(f"OCR შეცდომა: {e}")


def tcp_client():
    global tcp_socket, running, detection_enabled
    while running:
        try:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_socket.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
            log.info(f"TCP დაკავშირებული: {TCP_SERVER_IP}:{TCP_SERVER_PORT}")
            while running:
                data = tcp_socket.recv(1024)
                if not data:
                    break
                cmd = data.decode('utf-8').strip().upper()

                if cmd == "START":
                    with detection_lock:
                        detection_enabled = True
                    reset_all()
                    log.info(">>> DETECTION STARTED <<<")

                elif cmd == "STOP":
                    with detection_lock:
                        detection_enabled = False
                    handle_stop()
                    log.info(">>> DETECTION STOPPED <<<")

        except Exception as e:
            log.error(f"TCP კავშირი გაწყდა: {e}")
            if tcp_socket:
                tcp_socket.close()
            tcp_socket = None
            time.sleep(RECONNECT_DELAY)


def reset_all():
    for cam in cameras:
        with cam.data_lock:
            cam.last_boxes.clear()
            cam.last_seen.clear()
            cam.wagon_numbers.clear()
            cam.unique_numbers.clear()
            cam.next_display_id = 1
            cam.track_to_display.clear()
            cam.wagons_list.clear()
    log.info("ყველა გასუფთავდა (START)")


def handle_stop():
    all_wagons = []
    for cam in cameras:
        with cam.data_lock:
            for w in cam.wagons_list:
                entry = w.copy()
                entry.update({"channel": cam.idx, "camera": cam.name, "timestamp": datetime.now().isoformat()})
                all_wagons.append(entry)

    if all_wagons:
        try:
            with open(UNIQUE_WAGON_JSON, "w", encoding="utf-8") as f:
                json.dump(all_wagons, f, ensure_ascii=False, indent=2)
            log.info(f"JSON შენახული: {len(all_wagons)} ვაგონი")
        except Exception as e:
            log.error(f"JSON შენახვა ვერ მოხერხდა: {e}")

        if tcp_socket and hasattr(tcp_socket, 'fileno') and tcp_socket.fileno() != -1:
            try:
                msg = json.dumps(all_wagons, ensure_ascii=False) + "\n"
                tcp_socket.sendall(msg.encode("utf-8"))
                log.info(f"TCP: გაიგზავნა {len(all_wagons)} ვაგონი")
            except Exception as e:
                log.error(f"TCP გაგზავნა ვერ მოხერხდა: {e}")

    reset_all()


def display():
    if HEADLESS:
        while True:
            if not globals().get('running', False):
                break
            time.sleep(1)
        return

    while True:
        # ასე არასდროს იქნება SyntaxError ან UnboundLocalError
        if not globals().get('running', False):
            break

        canvas = np.zeros((900, 1940, 3), np.uint8)
        canvas[:] = (30, 30, 50)
        cv2.putText(canvas, "WAGON TRACKER v15.5 — სრულიად სტაბილური", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 3)

        with detection_lock:
            is_running = detection_enabled
        status = "მუშაობს" if is_running else "მზადაა (ელოდება START)"
        color = (0, 255, 0) if is_running else (0, 140, 255)
        cv2.putText(canvas, f"სტატუსი: {status}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

        total = sum(len(cam.wagons_list) for cam in cameras)
        cv2.putText(canvas, f"ჯამში: {total}", (20, 140), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 3)

        for i, cam in enumerate(cameras):
            with cam.frame_lock:
                if cam.latest_frame is None:
                    continue
                frame = cv2.resize(cam.latest_frame, (860, 480))
            x0 = 80 + i * 940
            canvas[200:680, x0:x0+860] = frame

            r = cam.cfg["roi"]
            h, w = cam.latest_frame.shape[:2]
            cv2.rectangle(canvas,
                (x0 + int(w * r["x1"] * 860/w), 200 + int(h * r["y1"] * 480/h)),
                (x0 + int(w * r["x2"] * 860/w), 200 + int(h * r["y2"] * 480/h)),
                (0, 0, 255), 6)

            cv2.putText(canvas, f"{cam.name} | {len(cam.wagons_list)}", (x0+20, 180),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

            if detection_enabled:
                with cam.data_lock:
                    for tid, (x1, y1, x2, y2) in cam.last_boxes.items():
                        sx = 860 / w
                        sy = 480 / h
                        color = (0, 255, 0) if tid in cam.wagon_numbers else (0, 165, 255)
                        label = f"{cam.track_to_display.get(tid, '?')}→{cam.wagon_numbers.get(tid, '?')}"
                        cv2.rectangle(canvas, (x0 + int(x1*sx), 200 + int(y1*sy)),
                                      (x0 + int(x2*sx), 200 + int(y2*sy)), color, 4)
                        cv2.putText(canvas, label, (x0 + int(x1*sx), 200 + int(y1*sy) - 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        cv2.imshow("WAGON TRACKER — FINAL v15.5", canvas)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            globals()['running'] = False
            break

    cv2.destroyAllWindows()


def main():
    global running, cameras

    signal.signal(signal.SIGINT, lambda s, f: globals().__setitem__("running", False))
    signal.signal(signal.SIGTERM, lambda s, f: globals().__setitem__("running", False))

    log.info("=== WAGON TRACKER INDUSTRIAL STARTED (v15.4) ===")

    Thread(target=ocr_worker, daemon=True).start()
    Thread(target=tcp_client, daemon=True).start()
    if ENABLE_GUI:
        Thread(target=display, daemon=True).start()

    cameras = [Camera(cfg, i) for i, cfg in enumerate(CAMERAS)]

    log.info("სისტემა მზადაა! გაგზავნე TCP-ით: START ან STOP")
    while running:
        time.sleep(1)

    log.info("=== სისტემა გაჩერდა ===")


if __name__ == "__main__":
    main()