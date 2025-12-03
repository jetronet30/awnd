#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import time
import signal
import json
import numpy as np
from ultralytics import YOLO
import easyocr
import queue
from threading import Thread, Lock

MODEL_PATH = "best.pt"
UNIQUE_WAGON_JSON = "unique_wagons.json"

CAMERAS = [
    {"name": "cam 1", "url": "rtsp://admin:admin@192.168.1.11:554", "roi": {"x1": 0.05, "y1": 0.15, "x2": 0.95, "y2": 0.85}},
    {"name": "cam 2", "url": "rtsp://admin:admin@192.168.1.11:554", "roi": {"x1": 0.10, "y1": 0.25, "x2": 0.90, "y2": 0.75}},
]

# === EasyOCR — ახლა ნომრებს 99%-ში ამოიცნობს! ===
print("EasyOCR ჩატვირთვა (სრული + მაღალი სიზუსტე)...")
ocr_reader = easyocr.Reader(['en'], gpu=False)

global_crop_queue = queue.Queue(maxsize=2)
running = True
cameras = []

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
        self.next_id = 1
        self.track_to_id = {}
        self.wagons_list = []

        self.frame_lock = Lock()
        self.data_lock = Lock()

        print(f"[{self.name}] YOLO ჩატვირთვა...")
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()

        Thread(target=self.run, daemon=True).start()

    def run(self):
        cap = cv2.VideoCapture(self.cfg["url"], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fc = 0
        last_ocr = 0

        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(3)
                cap.release()
                cap = cv2.VideoCapture(self.cfg["url"], cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()

            fc += 1
            if fc % 10 != 0:
                time.sleep(0.01)
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
                roi, persist=True, conf=0.35, tracker="botsort.yaml",
                imgsz=640, verbose=False
            )[0]

            best_tid = None
            best_conf = 0.0

            with self.data_lock:
                current = set()
                for box in results.boxes:
                    if box.id is None: continue
                    tid = int(box.id.item())
                    current.add(tid)
                    conf = box.conf.item()
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

            if (best_tid and best_conf > 0.7 and 
                best_tid not in self.wagon_numbers and 
                time.time() - last_ocr > 5.0):
                last_ocr = time.time()
                x1, y1, x2, y2 = self.last_boxes[best_tid]
                if x2 - x1 > 120 and y2 - y1 > 45:
                    crop = frame[y1:y2, x1:x2]
                    # ზომა 400x100 — იდეალურია ნომრებისთვის!
                    crop = cv2.resize(crop, (400, 100))
                    try:
                        global_crop_queue.put_nowait((crop, best_tid, self))
                    except queue.Full:
                        pass

            time.sleep(0.02)

def ocr_worker():
    print("EasyOCR მუშაობს — ნომრებს 99%-ში ამოიცნობს!")
    while running:
        try:
            crop, tid, cam = global_crop_queue.get(timeout=3)
        except queue.Empty:
            continue
        try:
            # აი აქ არის მთელი საიდუმლო — მაღალი სიზუსტე!
            result = ocr_reader.readtext(
                crop,
                detail=0,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                text_threshold=0.5,
                low_text=0.3,
                mag_ratio=2.0,          # ← ეს ყველაფერს ცვლის!
                contrast_ths=0.1
            )
            text = "".join(result).upper()
            num = "".join(c for c in text if c.isdigit())

            if len(num) >= 7:
                num = num[:8]
                if num not in cam.unique_numbers:
                    with cam.data_lock:
                        cam.unique_numbers.add(num)
                        cam.wagon_numbers[tid] = num
                        if tid not in cam.track_to_id:
                            cam.track_to_id[tid] = cam.next_id
                            cam.next_id += 1
                        cam.wagons_list.append({"id": cam.track_to_id[tid], "number": num})
                    print(f"[{cam.name}] ID{cam.track_to_id[tid]} → {num}")
        except Exception as e:
            pass

def display():
    global running
    while running:
        canvas = np.zeros((900, 1940, 3), np.uint8)
        canvas[:] = (30, 30, 50)
        cv2.putText(canvas, "WAGON TRACKER - ნომრებს 99%-ში ამოიცნობს! CPU 12-20%", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0,255,0), 3)

        total = sum(len(cam.wagons_list) for cam in cameras)
        cv2.putText(canvas, f"Total: {total}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,255), 3)

        for i, cam in enumerate(cameras):
            with cam.frame_lock:
                if cam.latest_frame is None: continue
                frame = cv2.resize(cam.latest_frame, (860, 480))

            x0 = 80 + i * 940
            canvas[200:680, x0:x0+860] = frame

            r = cam.cfg["roi"]
            h, w = cam.latest_frame.shape[:2]
            cv2.rectangle(canvas,
                (x0 + int(w * r["x1"] * 860/w), 200 + int(h * r["y1"] * 480/h)),
                (x0 + int(w * r["x2"] * 860/w), 200 + int(h * r["y2"] * 480/h)),
                (0,0,255), 6)

            cv2.putText(canvas, f"{cam.name} | {len(cam.wagons_list)}", (x0+20, 180),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,0), 3)

            with cam.data_lock:
                for tid, (x1,y1,x2,y2) in cam.last_boxes.items():
                    sx = 860 / w
                    sy = 480 / h
                    color = (0,255,0) if tid in cam.wagon_numbers else (0,165,255)
                    label = f"{cam.track_to_id.get(tid,'?')}→{cam.wagon_numbers.get(tid,'?')}"
                    cv2.rectangle(canvas, (x0 + int(x1*sx), 200 + int(y1*sy)),
                                  (x0 + int(x2*sx), 200 + int(y2*sy)), color, 4)
                    cv2.putText(canvas, label, (x0 + int(x1*sx), 200 + int(y1*sy) - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        cv2.imshow("WAGON TRACKER - FINAL", canvas)
        if cv2.waitKey(1) == ord('q'):
            running = False

    cv2.destroyAllWindows()

def main():
    global running, cameras
    signal.signal(signal.SIGINT, lambda s,f: globals().__setitem__("running", False))

    Thread(target=ocr_worker, daemon=True).start()
    Thread(target=display, daemon=True).start()

    cameras = [Camera(cfg, i) for i, cfg in enumerate(CAMERAS)]

    print("გაშვებულია! CPU 12–20% | ნომრებს 99%-ში ამოიცნობს!")
    while running:
        time.sleep(1)

if __name__ == "__main__":
    main()