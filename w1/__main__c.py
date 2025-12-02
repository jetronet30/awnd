#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import subprocess
import os
import threading
import time
import signal
import sys
import torch
import re
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import queue

# ================================
# კონფიგურაცია
# ================================
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "./streams/cam1"
HLS_PLAYLIST = os.path.join(OUTPUT_DIR, "index.m3u8")
LOG_FILE = "wagon_ocr_results.txt"

GUI_WINDOW_NAME = "WAGON TRACKER - LIVE"
GUI_WIDTH, GUI_HEIGHT = 1280, 720

MIN_CONFIDENCE_OCR = 0.45
MATCH_THRESHOLD = 280
MIN_CONFIDENCE_FOR_ID = 0.35

# ================================
# გლობალური ცვლადები
# ================================
running = True
frame_queue = queue.Queue(maxsize=3)
crop_queue = queue.Queue(maxsize=3)
ffmpeg_process = None
model = None
cap = None
last_ocr_text = "wagon: -"
ocr_lock = threading.Lock()
gui_frame = None
gui_lock = threading.Lock()

known_sectors = {}
next_id = 1

stats = {
    'total_frames': 0, 'yolo_runs': 0, 'ocr_attempts': 0, 'successful_ocr': 0,
    'ffmpeg_frames': 0, 'gui_frames': 0, 'start_time': time.time()
}

# ================================
# დამხმარე ფუნქციები
# ================================
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_stable_id(center, conf):
    global next_id
    if conf < MIN_CONFIDENCE_FOR_ID:
        return None

    best_id = None
    best_dist = float('inf')

    for sid, known in list(known_sectors.items()):
        dist = ((center[0] - known[0])**2 + (center[1] - known[1])**2)**0.5
        if dist < best_dist and dist < MATCH_THRESHOLD:
            best_dist = dist
            best_id = sid

    if best_id is not None:
        # გლუვი განახლება
        old = known_sectors[best_id]
        known_sectors[best_id] = (
            int(old[0] * 0.9 + center[0] * 0.1),
            int(old[1] * 0.9 + center[1] * 0.1)
        )
        return best_id
    else:
        new_id = next_id
        known_sectors[new_id] = center
        next_id += 1
        # ძველი ID-ების გასუფთავება
        if len(known_sectors) > 60:
            oldest = min(known_sectors.keys())
            known_sectors.pop(oldest, None)
        return new_id

# ================================
# OCR Thread
# ================================
def ocr_worker():
    global last_ocr_text, running
    print(" [OCR] TrOCR მოდელის ჩატვირთვა...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model.to(device)
    trocr_model.eval()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n=== სესია: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print(f" [OCR] მზად ({device})")

    while running:
        try:
            cropped, wagon_id = crop_queue.get(timeout=1.0)
            if cropped is None:
                break

            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            pil = pil.resize((384, 96), Image.LANCZOS)

            pixel_values = processor(pil, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = trocr_model.generate(pixel_values, max_new_tokens=12, num_beams=4)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            numbers = re.sub(r'[^\d]', '', text)

            if len(numbers) >= 6:
                result = f"wagon-{wagon_id}: {numbers}"
                with ocr_lock:
                    last_ocr_text = result
                stats['successful_ocr'] += 1
                print(f" [OCR] {result}")

                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {result}\n")

        except queue.Empty:
            continue
        except Exception as e:
            if running:
                print(f" [OCR] შეცდომა: {e}")

# ================================
# GUI Thread
# ================================
def gui_thread():
    global running, gui_frame
    cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(GUI_WINDOW_NAME, GUI_WIDTH, GUI_HEIGHT)
    print(" [GUI] გახსნილია (ESC = გასვლა)")

    while running:
        with gui_lock:
            if gui_frame is None:
                time.sleep(0.01)
                continue
            disp = gui_frame.copy()

        with ocr_lock:
            ocr_txt = last_ocr_text if "wagon" in last_ocr_text else "OCR: მუშაობს..."

        cv2.putText(disp, ocr_txt, (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 3)
        cv2.putText(disp, f"ვაგონები: {next_id-1}", (20, 140), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)

        uptime = time.time() - stats['start_time']
        gui_fps = stats['gui_frames'] / max(uptime, 0.1)
        cv2.putText(disp, f"FPS: {gui_fps:.1f}", (20, 190), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow(GUI_WINDOW_NAME, disp)
        stats['gui_frames'] += 1

        if cv2.waitKey(1) == 27:  # ESC
            running = False
            break

    cv2.destroyAllWindows()
    print(" [GUI] დახურულია")

# ================================
# RTSP Reader
# ================================
def rtsp_reader():
    global cap, running
    while running:
        if not cap or not cap.isOpened():
            print(" [RTSP] ხელახლა დაკავშირება...")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(2)

        ret, frame = cap.read()
        if ret and frame is not None:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try: frame_queue.get_nowait()
                except: pass
                frame_queue.put_nowait(frame)
        else:
            cap.release()
            cap = None
            time.sleep(1)

# ================================
# FFMPEG HLS
# ================================
def start_ffmpeg(width, height):
    global ffmpeg_process
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".ts") or f == "index.m3u8":
            try: os.remove(os.path.join(OUTPUT_DIR, f))
            except: pass

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", "25", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-g", "50", "-bf", "0", "-f", "hls",
        "-hls_time", "2", "-hls_list_size", "5",
        "-hls_flags", "delete_segments+append_list",
        "-hls_segment_filename", os.path.join(OUTPUT_DIR, "seg%04d.ts"),
        HLS_PLAYLIST
    ]

    ffmpeg_process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print(f" [HLS] გაშვებული → {HLS_PLAYLIST}")

# ================================
# სტატუსი
# ================================
def print_status():
    while running:
        time.sleep(5)
        uptime = time.time() - stats['start_time']
        print(f"""
[INFO {datetime.now().strftime('%H:%M:%S')}]
   ვაგონები: {next_id-1} | OCR წარმატებული: {stats['successful_ocr']}
   FPS → HLS: {stats['ffmpeg_frames']/max(uptime,1):.1f} | GUI: {stats['gui_frames']/max(uptime,1):.1f}
   ბოლო OCR: {last_ocr_text}
   HLS: http://თქვენი_IP:8000/streams/cam1/index.m3u8
        """.strip())

# ================================
# გაწმენდა
# ================================
def cleanup():
    global running, ffmpeg_process, cap
    print("\n გაჩერება...")
    running = False
    time.sleep(0.5)

    crop_queue.put((None, None))
    if ffmpeg_process:
        try: ffmpeg_process.stdin.close()
        except: pass
        ffmpeg_process.terminate()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print(" დასრულდა!")

# ================================
# მთავარი ციკლი
# ================================
def main():
    global running, model, gui_frame

    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    # Threads
    threading.Thread(target=ocr_worker, daemon=True).start()
    threading.Thread(target=gui_thread, daemon=True).start()
    threading.Thread(target=rtsp_reader, daemon=True).start()
    threading.Thread(target=print_status, daemon=True).start()

    time.sleep(4)  # მოდელების ჩატვირთვა

    # YOLO
    print(" [YOLO] მოდელის ჩატვირთვა...")
    model = YOLO(MODEL_PATH)
    model.fuse()
    print(" [YOLO] მზად")

    # პირველი ფრეიმი → განსაზღვროთ ზომა
    frame = None
    for _ in range(50):
        try:
            frame = frame_queue.get(timeout=1)
            break
        except:
            continue
    if frame is None:
        print(" [ERROR] კამერასთან დაკავშირება ვერ მოხერხდა")
        return

    height, width = frame.shape[:2]
    start_ffmpeg(width, height)

    # ROI
    roi_x1, roi_y1 = int(width * 0.05), int(height * 0.1)
    roi_x2, roi_y2 = int(width * 0.95), int(height * 0.9)

    stats['start_time'] = time.time()
    print(f"\n სისტემა გაშვებული! {width}x{height} @ 25fps\n")

    frame_counter = 0
    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except:
            continue

        frame_counter += 1
        stats['total_frames'] += 1

        # YOLO ყოველ 16-18 ფრეიმში
        if frame_counter % 17 == 0:
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size > 0:
                results = model(roi, conf=0.25, imgsz=640, verbose=False)[0]
                best_box = None
                best_conf = 0
                best_id = None

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    gx1, gy1 = roi_x1 + x1, roi_y1 + y1
                    gx2, gy2 = roi_x1 + x2, roi_y1 + y2
                    center = get_center((gx1, gy1, gx2, gy2))
                    wagon_id = get_stable_id(center, conf)

                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{wagon_id}", (gx1, gy1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if conf > best_conf and conf > MIN_CONFIDENCE_OCR:
                        best_conf = conf
                        best_box = (gx1, gy1, gx2, gy2)
                        best_id = wagon_id

                # OCR ყოველ ~1.5 წამში
                if best_box and frame_counter % 34 == 0:
                    crop = frame[best_box[1]:best_box[3], best_box[0]:best_box[2]]
                    if crop.size > 0 and crop_queue.qsize() < 2:
                        crop_queue.put((crop.copy(), best_id))

        # HLS გაგზავნა
        if ffmpeg_process and ffmpeg_process.poll() is None:
            try:
                out = frame.copy()
                cv2.rectangle(out, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 3)
                cv2.putText(out, f"ვაგონები: {next_id-1}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 0), 3)
                with ocr_lock:
                    cv2.putText(out, last_ocr_text, (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                ffmpeg_process.stdin.write(out.tobytes())
                stats['ffmpeg_frames'] += 1
            except:
                pass

        # GUI
        try:
            scale = min(GUI_WIDTH/width, GUI_HEIGHT/height)
            resized = cv2.resize(frame, (int(width*scale), int(height*scale)))
            bg = np.zeros((GUI_HEIGHT, GUI_WIDTH, 3), np.uint8)
            xoff = (GUI_WIDTH - resized.shape[1]) // 2
            yoff = (GUI_HEIGHT - resized.shape[0]) // 2
            bg[yoff:yoff+resized.shape[0], xoff:xoff+resized.shape[1]] = resized
            with gui_lock:
                gui_frame = bg
        except:
            pass

    cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()