#!/usr/bin/env python3
import cv2
import os
import threading
import time
import signal
import sys
import json
import torch
import numpy as np
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

MIN_CONFIDENCE_OCR = 0.6
TCP_SERVER_IP = "192.168.1.81"
TCP_SERVER_PORT = 45000
TCP_RECONNECT_DELAY = 5
WAGON_NUMBER_LENGTH = 8

# ================================
# გლობალური ცვლადები
# ================================
crop_queue = queue.Queue(maxsize=2)
command_queue = queue.Queue()
running = True

model = None
cap = None
tcp_socket = None
ocr_lock = threading.Lock()

last_boxes = {}
cached_wagons = {}
wagon_numbers = {}
unique_numbers = set()

display_id_counter = 1
track_to_display_id = {}

wagons_data = {
    "session": {"start_time": datetime.now().isoformat(), "total_wagons": 0, "unique_numbers": 0},
    "wagons": [],
    "ocr_results": []
}

frame_count = 0
g_conf = "0.000"

# ================================
# JSON ლოგირება
# ================================
def log_unique(track_id, number):
    global display_id_counter
    if track_id not in track_to_display_id:
        track_to_display_id[track_id] = display_id_counter
        display_id_counter += 1
    
    display_id = track_to_display_id[track_id]
    
    wagons_data["wagons"].append({
        "id": display_id,
        "number": number
    })
    wagons_data["session"]["unique_numbers"] += 1
    
    print(f"ახალი უნიკალური: {display_id}-->{number}")

def save_logs():
    simple_list = wagons_data["wagons"]
    with open(UNIQUE_WAGON_JSON, "w", encoding="utf-8") as f:
        json.dump(simple_list, f, ensure_ascii=False, indent=2)

# ================================
# OCR Worker
# ================================
def ocr_worker():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    trocr_model.eval()

    while running:
        try:
            cropped, track_id = crop_queue.get(timeout=0.3)
            if cropped is None:
                break

            pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).resize((224, 56), Image.BILINEAR)
            pixel_values = processor(pil, return_tensors="pt").pixel_values

            with torch.no_grad():
                ids = trocr_model.generate(pixel_values, max_length=12, num_beams=1)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            cleaned = ''.join(filter(str.isdigit, text))

            valid = len(cleaned) == WAGON_NUMBER_LENGTH

            if valid and cleaned not in unique_numbers and track_id not in wagon_numbers:
                unique_numbers.add(cleaned)
                wagon_numbers[track_id] = cleaned
                log_unique(track_id, cleaned)

        except queue.Empty:
            continue
        except Exception as e:
            if running:
                print(f"OCR Error: {e}")

# ================================
# TCP + სესია
# ================================
def tcp_client_thread():
    global running, tcp_socket
    while running:
        try:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"ვუკავშირდებით {TCP_SERVER_IP}:{TCP_SERVER_PORT}...")
            tcp_socket.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
            print("TCP დაკავშირებულია!")
            while running:
                data = tcp_socket.recv(1024)
                if not data:
                    break
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["START", "STOP"]:
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
    global display_id_counter, track_to_display_id
    with ocr_lock:
        last_boxes.clear()
        cached_wagons.clear()
        wagon_numbers.clear()
        unique_numbers.clear()
        display_id_counter = 1
        track_to_display_id.clear()
        wagons_data["wagons"] = []
        wagons_data["ocr_results"] = []
        wagons_data["session"] = {
            "start_time": datetime.now().isoformat(),
            "total_wagons": 0,
            "unique_numbers": 0
        }
    print("სესია გადატვირთულია! (START) — ყველაფერი გასუფთავდა.")

def handle_stop_command():
    print("STOP მიღებულია — ვაგზავნით სუფთა JSON-ს...")
    save_logs()

    if not wagons_data["wagons"]:
        print("ცარიელი სია — არაფერი გასაგზავნი")
        return

    if tcp_socket is None:
        print("TCP კავშირი არ არის!")
        return

    try:
        # ლამაზი JSON ფორმატით (indent=2)
        json_str = json.dumps(wagons_data["wagons"], ensure_ascii=False, indent=2)

        # გაგზავნა მხოლოდ JSON + \n ბოლოს
        tcp_socket.sendall((json_str + "\n").encode('utf-8'))

        print(f"გაიგზავნა {len(wagons_data['wagons'])} ვაგონი:")
        print(json_str)

        reset_session()

    except Exception as e:
        print(f"შეცდომა გაგზავნისას: {e}")

# ================================
# მთავარი ციკლი
# ================================
def main():
    global running, model, cap, frame_count, g_conf

    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    Thread(target=tcp_client_thread, daemon=True).start()
    Thread(target=ocr_worker, daemon=True).start()

    model = YOLO(MODEL_PATH)
    model.fuse()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0:
        print("კამერა არ მუშაობს!")
        return

    roi = (int(width*0.10), int(width*0.90), int(height*0.20), int(height*0.80))
    rx1_roi, rx2_roi, ry1_roi, ry2_roi = roi

    cv2.namedWindow("WAGON TRACKER - UNIQUE NUMBERS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WAGON TRACKER - UNIQUE NUMBERS", 1280, 720)

    while running:
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd == "START":
                    reset_session()
                elif cmd == "STOP":
                    handle_stop_command()
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if not ret:
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

        cv2.rectangle(disp,
            (int(rx1_roi*scale)+offset_x, int(ry1_roi*scale)+offset_y),
            (int(rx2_roi*scale)+offset_x, int(ry2_roi*scale)+offset_y),
            (0, 0, 255), 2)

        best_box = None
        best_conf = 0
        best_id = None

        if frame_count % 3 == 0:
            roi_frame = frame[ry1_roi:ry2_roi, rx1_roi:rx2_roi]

            results = model.track(
                source=roi_frame,
                conf=0.25,
                iou=0.6,
                imgsz=640,
                tracker="botsort.yaml",
                persist=True,
                verbose=False
            )[0]

            for box in results.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                conf = box.conf.item()
                g_conf = f"{conf:.3f}"

                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                gx1 = rx1_roi + bx1
                gy1 = ry1_roi + by1
                gx2 = rx1_roi + bx2
                gy2 = ry1_roi + by2

                last_boxes[track_id] = (gx1, gy1, gx2, gy2)
                cached_wagons[track_id] = frame_count

                if conf > best_conf and conf >= MIN_CONFIDENCE_OCR:
                    best_conf = conf
                    best_box = (gx1, gy1, gx2, gy2)
                    best_id = track_id

            for tid in list(cached_wagons):
                if frame_count - cached_wagons[tid] > 45:
                    cached_wagons.pop(tid, None)
                    last_boxes.pop(tid, None)

        if best_box and frame_count % 12 == 0:
            x1, y1, x2, y2 = best_box
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop_queue.qsize() < 2:
                crop_queue.put((crop.copy(), best_id))

        for track_id, (x1, y1, x2, y2) in last_boxes.items():
            if track_id not in cached_wagons:
                continue

            display_id = track_to_display_id.get(track_id, "?")
            dx1 = int(x1 * scale) + offset_x
            dy1 = int(y1 * scale) + offset_y
            dx2 = int(x2 * scale) + offset_x
            dy2 = int(y2 * scale) + offset_y

            color = (0, 165, 255)
            if track_id in wagon_numbers:
                color = (0, 255, 0)
            elif best_id == track_id:
                color = (0, 255, 255)

            cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), color, 3)
            label = f"{display_id}-->{wagon_numbers.get(track_id, '?')}"
            cv2.putText(disp, label, (dx1, dy1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        cv2.putText(disp, f"Conf: {g_conf}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(disp, f"Unique: {wagons_data['session']['unique_numbers']}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        cv2.imshow("WAGON TRACKER - UNIQUE NUMBERS", disp)
        if cv2.waitKey(1) == 27:
            break

    cleanup()

def cleanup():
    global running
    print("\nპროგრამა მთავრდება...")
    running = False
    time.sleep(1)
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()