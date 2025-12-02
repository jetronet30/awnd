#!/usr/bin/env python3
import cv2
import os
import threading
import time
import signal
import sys
import torch
import re
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import queue
import socket
from threading import Thread

# ================================
# კონფიგურაცია
# ================================
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"
MODEL_PATH = "best.pt"

UNIQUE_WAGON_JSON = "unique_wagons.json"
ALL_OCR_JSON = "all_ocr_results.json"

MIN_CONFIDENCE_OCR = 0.8
MIN_CONFIDENCE_FOR_ID = 0.8
WAGON_NUMBER_LENGTH = 8

TCP_SERVER_IP = "127.0.0.1"
TCP_SERVER_PORT = 5000
TCP_RECONNECT_DELAY = 5

# ================================
# გლობალური ცვლადები
# ================================
crop_queue = queue.Queue(maxsize=2)
command_queue = queue.Queue()
running = True
detection_active = False

model = None
cap = None
tcp_socket = None
last_valid_ocr_text = "waiting..."
ocr_lock = threading.Lock()

g_conf = '0.0'

wagons_data = {
    "session": {"start_time": datetime.now().isoformat(), "total_wagons": 0, "unique_numbers": 0},
    "wagons": [],
    "ocr_results": []
}

wagon_centers = {}      # (x, y) ცენტრები
wagon_numbers = {}      # ID → ნომერი
unique_numbers = set()
next_id = 1
cached_wagons = {}      # ID → ბოლო ფრეიმის ნომერი (სიცოცხლისთვის)
last_boxes = {}         # <<< ახალი: ID → ბოლო ნამდვილი bounding box (x1,y1,x2,y2)
frame_count = 0

# ================================
# JSON ლოგირება
# ================================
def save_logs():
    with open(UNIQUE_WAGON_JSON, "w", encoding="utf-8") as f:
        json.dump({"wagons": wagons_data["wagons"]}, f, ensure_ascii=False, indent=2)
    
    full = wagons_data.copy()
    full["session"]["end_time"] = datetime.now().isoformat()
    with open(ALL_OCR_JSON, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

def log_unique(wagon_id, number):
    wagons_data["wagons"].append({"id": wagon_id, "number": number})
    wagons_data["session"]["unique_numbers"] += 1

def log_ocr(wagon_id, raw, cleaned, valid):
    wagons_data["ocr_results"].append({
        "timestamp": datetime.now().isoformat(),
        "wagon_id": wagon_id,
        "raw_text": raw,
        "cleaned": cleaned,
        "length": len(cleaned),
        "valid": valid
    })

# ================================
# ტრეკინგი
# ================================
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_wagon_id(center, conf, box=None):
    global next_id
    if conf < MIN_CONFIDENCE_FOR_ID:
        return None

    best_id = None
    best_dist = float("inf")
    for wid, old in wagon_centers.items():
        dist = ((center[0] - old[0])**2 + (center[1] - old[1])**2)**0.5
        if dist < 200 and dist < best_dist:
            best_dist = dist
            best_id = wid

    if best_id is not None:
        old = wagon_centers[best_id]
        wagon_centers[best_id] = (int(old[0]*0.7 + center[0]*0.3), int(old[1]*0.7 + center[1]*0.3))
        if box is not None:
            last_boxes[best_id] = box
        return best_id
    else:
        wagon_centers[next_id] = center
        wagon_numbers[next_id] = None
        if box is not None:
            last_boxes[next_id] = box
        wagons_data["session"]["total_wagons"] += 1
        next_id += 1
        return next_id - 1

# ================================
# OCR Worker
# ================================
def ocr_worker():
    global last_valid_ocr_text, running
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    trocr_model.eval()

    while running:
        try:
            cropped, wagon_id = crop_queue.get(timeout=0.3)
            if cropped is None:
                break

            pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).resize((224, 56), Image.BILINEAR)
            pixel_values = processor(pil, return_tensors="pt").pixel_values

            with torch.no_grad():
                ids = trocr_model.generate(pixel_values, max_length=12, num_beams=1)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            cleaned = re.sub(r"[^0-9]", "", text)

            valid = len(cleaned) == WAGON_NUMBER_LENGTH
            log_ocr(wagon_id, text, cleaned, valid)

            if valid and cleaned not in unique_numbers and wagon_numbers[wagon_id] is None:
                unique_numbers.add(cleaned)
                wagon_numbers[wagon_id] = cleaned
                log_unique(wagon_id, cleaned)
                last_valid_ocr_text = f"{wagon_id}-->{cleaned}"
                print(f"ახალი უნიკალური: {wagon_id}-->{cleaned}")
            elif wagon_numbers.get(wagon_id):
                last_valid_ocr_text = f"{wagon_id}-->{wagon_numbers[wagon_id]}"

        except queue.Empty:
            continue
        except Exception as e:
            if running:
                print(f"OCR Error: {e}")

# ================================
# TCP კლიენტი + სესია
# ================================
def tcp_client_thread():
    global running, tcp_socket
    while running:
        try:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"ვუკავშირდებით სერვერს {TCP_SERVER_IP}:{TCP_SERVER_PORT}...")
            tcp_socket.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
            print("TCP დაკავშირებულია!")

            while running:
                data = tcp_socket.recv(1024)
                if not data:
                    print("სერვერი გათიშულია")
                    break
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["START", "STOP"]:
                    print(f"მიღებული ბრძანება: {cmd}")
                    command_queue.put(cmd)

        except Exception as e:
            if running:
                print(f"TCP შეცდომა: {e}")
        finally:
            if tcp_socket:
                tcp_socket.close()
                tcp_socket = None
        if running:
            time.sleep(TCP_RECONNECT_DELAY)

def reset_session():
    global next_id, wagon_centers, wagon_numbers, unique_numbers, cached_wagons, last_boxes
    global wagons_data, last_valid_ocr_text, detection_active

    with ocr_lock:
        next_id = 1
        wagon_centers = {}
        wagon_numbers = {}
        unique_numbers = set()
        cached_wagons = {}
        last_boxes = {}                    # <<< გავასუფთავოთ
        last_valid_ocr_text = "waiting..."

        wagons_data = {
            "session": {"start_time": datetime.now().isoformat(), "total_wagons": 0, "unique_numbers": 0},
            "wagons": [], "ocr_results": []
        }
    print("სესია გადატვირთულია! დეტექცია ჩართულია (START)")

def handle_stop_command():
    global detection_active
    print("STOP ბრძანება – სესია მთავრდება...")
    save_logs()
    try:
        if os.path.exists(UNIQUE_WAGON_JSON):
            with open(UNIQUE_WAGON_JSON, 'r', encoding='utf-8') as f:
                json_data = f.read()
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            send_sock.settimeout(10)
            send_sock.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
            data_bytes = json_data.encode('utf-8')
            length_prefix = f"{len(data_bytes):08d}".encode('utf-8')
            send_sock.sendall(length_prefix + data_bytes)
            print(f"JSON გაგზავნილია სერვერზე ({len(data_bytes)} ბაიტი)")
            send_sock.close()
    except Exception as e:
        print(f"JSON-ის გაგზავნის შეცდომა: {e}")
    detection_active = False
    print("სესია დასრულებული. ველოდებით ახალ START-ს...")

# ================================
# მთავარი ციკლი
# ================================
def main():
    global running, model, cap, frame_count, detection_active, g_conf

    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    Thread(target=tcp_client_thread, daemon=True).start()
    Thread(target=ocr_worker, daemon=True).start()

    model = YOLO(MODEL_PATH)
    model.fuse()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0:
        print("კამერა არ მუშაობს")
        return

    roi = (int(width*0.05), int(width*0.95), int(height*0.10), int(height*0.90))
    x1, x2, y1, y2 = roi

    cv2.namedWindow("WAGON TRACKER - UNIQUE NUMBERS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WAGON TRACKER - UNIQUE NUMBERS", 1280, 720)

    while running:
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd == "START":
                    reset_session()
                    detection_active = True
                elif cmd == "STOP":
                    handle_stop_command()
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        scale = min(1280/width, 720/height)
        disp = cv2.resize(frame, (int(width*scale), int(height*scale)))
        bg = np.zeros((720, 1280, 3), np.uint8)
        offset_y = (720 - disp.shape[0]) // 2
        offset_x = (1280 - disp.shape[1]) // 2
        bg[offset_y:offset_y+disp.shape[0], offset_x:offset_x+disp.shape[1]] = disp
        disp = bg

        frame_count += 1

        # ROI ჩარჩო
        rx1 = int(x1*scale) + offset_x
        ry1 = int(y1*scale) + offset_y
        rx2 = int(x2*scale) + offset_x
        ry2 = int(y2*scale) + offset_y
        cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        best_box = None
        best_conf = 0
        best_id = None

        # === დეტექცია ყოველ მესამე ფრეიმზე ===
        if frame_count % 3 == 0:
            roi_frame = frame[y1:y2, x1:x2]
            results = model(roi_frame, conf=0.2, imgsz=320, verbose=False)[0]

            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                g_conf = f"{conf:.3f}"

                gx1, gy1 = x1 + bx1, y1 + by1
                gx2, gy2 = x1 + bx2, y1 + by2
                center = get_center((gx1, gy1, gx2, gy2))
                wid = get_wagon_id(center, conf, box=(gx1, gy1, gx2, gy2))

                if wid is not None:
                    cached_wagons[wid] = frame_count

                    if conf > best_conf and conf >= MIN_CONFIDENCE_OCR:
                        best_conf = conf
                        best_box = (gx1, gy1, gx2, gy2)
                        best_id = wid

            # წაშალოთ მკვდარი ტრეკები
            for wid in list(cached_wagons):
                if frame_count - cached_wagons[wid] > 30:
                    del cached_wagons[wid]
                    if wid in last_boxes:
                        del last_boxes[wid]

        # === OCR კროპი ===
        if best_box and frame_count % 15 == 0:
            x1b, y1b, x2b, y2b = best_box
            crop = frame[y1b:y2b, x1b:x2b]
            if crop.size > 0 and crop_queue.qsize() < 2:
                crop_queue.put((crop.copy(), best_id), block=False)

        # === ზუსტი ხატვა — ყოველ ფრეიმზე, რეალური box-ებით ===
        for wid, center in wagon_centers.items():
            if wid not in last_boxes:
                continue
            if frame_count - cached_wagons.get(wid, 0) > 50:
                continue

            x1b, y1b, x2b, y2b = last_boxes[wid]

            dx1 = int(x1b * scale) + offset_x
            dy1 = int(y1b * scale) + offset_y
            dx2 = int(x2b * scale) + offset_x
            dy2 = int(y2b * scale) + offset_y

            color = (0, 165, 255)  # ნარინჯისფერი
            if wagon_numbers.get(wid):
                color = (0, 255, 0)    # მწვანე — ამოცნობილი
            elif best_id == wid and best_conf >= MIN_CONFIDENCE_OCR:
                color = (0, 255, 255)  # ყვითელი — OCR კანდიდატი

            cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), color, 3)
            label = f"{wid}-->{wagon_numbers.get(wid, '?')}"
            cv2.putText(disp, label, (dx1, dy1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

            # ცენტრში პატარა წრე
            cx = int(center[0] * scale) + offset_x
            cy = int(center[1] * scale) + offset_y
            cv2.circle(disp, (cx, cy), 12, color, -1)

        # === სტატუსი ===
        cv2.putText(disp, f"Conf: {g_conf}", (400, 50), cv2.FONT_ITALIC, 1.2, (0, 255, 0), 2)
        cv2.putText(disp, f"wagons: {wagons_data['session']['unique_numbers']}", (20, 50),
                    cv2.FONT_ITALIC, 1.2, (0, 255, 0), 2)

        cv2.imshow("WAGON TRACKER - UNIQUE NUMBERS", disp)
        if cv2.waitKey(1) == 27:
            break

    cleanup()

# ================================
# გაწმენდა
# ================================
def cleanup():
    global running
    print("\nგაჩერება...")
    running = False
    time.sleep(1)
    save_logs()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()