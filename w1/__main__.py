#!/usr/bin/env python3
import cv2
import os
import time
import signal
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
import subprocess

# ================================
# კონფიგურაცია
# ================================
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"
MODEL_PATH = "best.pt"

UNIQUE_WAGON_JSON = "unique_wagons.json"
MIN_CONFIDENCE_OCR = 0.7
TCP_SERVER_IP = "127.0.0.1"
TCP_SERVER_PORT = 45002
WAGON_NUMBER_LENGTH = 8

# HLS
HLS_DIR = "hls"
HLS_PLAYLIST = os.path.join(HLS_DIR, "index.m3u8")
os.makedirs(HLS_DIR, exist_ok=True)

# ================================
# გლობალურები
# ================================
crop_queue = queue.Queue(maxsize=2)
command_queue = queue.Queue()
running = True

model = None
cap = None
tcp_socket = None
ffmpeg_process = None

last_boxes = {}
cached_wagons = {}
wagon_numbers = {}
unique_numbers = set()

display_id_counter = 1
track_to_display_id = {}

wagons_data = {
    "session": {"start_time": datetime.now().isoformat(), "total_wagons": 0, "unique_numbers": 0},
    "wagons": []
}

frame_count = 0
g_conf = "0.000"

detection_active = False  # ← ახლა START/STOP აკონტროლებს ამას

# ================================
# HLS სტრიმინგი
# ================================
def start_hls_stream():
    global ffmpeg_process
    if ffmpeg_process:
        return

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1280x720",
        "-r", "25",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-g", "50",
        "-keyint_min", "25",
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "100",
        "-hls_flags", "delete_segments",
        "-hls_segment_filename", os.path.join(HLS_DIR, "seg%03d.ts"),
        HLS_PLAYLIST
    ]

    print("HLS სტრიმინგი იწყება...")
    ffmpeg_process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ================================
# დანარჩენი ფუნქციები
# ================================
def log_unique(track_id, number):
    global display_id_counter
    if track_id not in track_to_display_id:
        track_to_display_id[track_id] = display_id_counter
        display_id_counter += 1
    display_id = track_to_display_id[track_id]
    wagons_data["wagons"].append({"id": display_id, "number": number})
    wagons_data["session"]["unique_numbers"] += 1
    print(f"ახალი უნიკალური: {display_id} --> {number}")

def save_logs():
    with open(UNIQUE_WAGON_JSON, "w", encoding="utf-8") as f:
        json.dump(wagons_data["wagons"], f, ensure_ascii=False, indent=2)

def ocr_worker():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    trocr_model.eval()

    while running:
        try:
            cropped, track_id = crop_queue.get(timeout=0.3)
            pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).resize((224, 56))
            pixel_values = processor(pil, return_tensors="pt").pixel_values
            with torch.no_grad():
                ids = trocr_model.generate(pixel_values, max_length=12)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            cleaned = ''.join(c for c in text if c.isdigit())

            if (len(cleaned) == WAGON_NUMBER_LENGTH and
                cleaned not in unique_numbers and
                track_id not in wagon_numbers):
                unique_numbers.add(cleaned)
                wagon_numbers[track_id] = cleaned
                log_unique(track_id, cleaned)

        except queue.Empty:
            continue
        except Exception as e:
            if running:
                print(f"OCR Error:", e)

def tcp_client_thread():
    global tcp_socket, running
    while running:
        try:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_socket.connect((TCP_SERVER_IP, TCP_SERVER_PORT))
            print("TCP დაკავშირებულია")
            while running:
                data = tcp_socket.recv(1024)
                if not data: break
                cmd = data.decode().strip().upper()
                if cmd in ["START", "STOP"]:
                    command_queue.put(cmd)
        except Exception as e:
            if running: print("TCP შეცდომა:", e)
        finally:
            if tcp_socket:
                tcp_socket.close()
                tcp_socket = None
        if running:
            time.sleep(5)

def reset_session():
    global display_id_counter, track_to_display_id
    last_boxes.clear()
    cached_wagons.clear()
    wagon_numbers.clear()
    unique_numbers.clear()
    display_id_counter = 1
    track_to_display_id.clear()
    wagons_data["wagons"].clear()
    wagons_data["session"]["unique_numbers"] = 0
    wagons_data["session"]["start_time"] = datetime.now().isoformat()

def handle_stop_command():
    print("STOP მიღებულია — ვაგზავნი JSON-ს")
    save_logs()
    if tcp_socket and wagons_data["wagons"]:
        try:
            json_str = json.dumps(wagons_data["wagons"], ensure_ascii=False)
            tcp_socket.sendall((json_str + "\n").encode('utf-8'))
            print(f"გაიგზავნა {len(wagons_data['wagons'])} ვაგონი")
        except Exception as e:
            print("გაგზავნა ვერ მოხერხდა:", e)
    reset_session()

# ================================
# მთავარი ციკლი
# ================================
def main():
    global running, model, cap, frame_count, g_conf, ffmpeg_process, detection_active

    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    Thread(target=tcp_client_thread, daemon=True).start()
    Thread(target=ocr_worker, daemon=True).start()

    model = YOLO(MODEL_PATH)
    model.fuse()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0:
        print("კამერა არ პასუხობს!")
        return

    roi = (int(w*0.005), int(w*0.995), int(h*0.20), int(h*0.80))
    rx1, rx2, ry1, ry2 = roi

    cv2.namedWindow("WAGON TRACKER", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WAGON TRACKER", 1280, 720)

    start_hls_stream()

    while running:
        # ბრძანებების დამუშავება
        try:
            while True:
                cmd = command_queue.get_nowait()
                if cmd == "START":
                    reset_session()
                    detection_active = True
                    print("დეტექცია გააქტიურდა (START)")
                elif cmd == "STOP":
                    handle_stop_command()
                    detection_active = False
                    print("დეტექცია გაჩერდა (STOP)")
        except queue.Empty:
            pass

        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        scale = min(1280/w, 720/h)
        resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
        disp = np.zeros((720, 1280, 3), np.uint8)
        yo = (720 - resized.shape[0]) // 2
        xo = (1280 - resized.shape[1]) // 2
        disp[yo:yo+resized.shape[0], xo:xo+resized.shape[1]] = resized

        frame_count += 1

        # ROI ჩარჩო
        cv2.rectangle(disp,
            (int(rx1*scale)+xo, int(ry1*scale)+yo),
            (int(rx2*scale)+xo, int(ry2*scale)+yo),
            (0, 0, 255), 3)

        best_box = None
        best_conf = 0
        best_id = None

        # გაწმენდა მკვდარი ტრეკების
        for tid in list(cached_wagons):
            if frame_count - cached_wagons[tid] > 45:
                cached_wagons.pop(tid, None)
                last_boxes.pop(tid, None)

        # დეტექცია მხოლოდ აქტიურ რეჟიმში
        if detection_active and frame_count % 3 == 0:
            roi_frame = frame[ry1:ry2, rx1:rx2]
            results = model.track(roi_frame, conf=0.25, iou=0.6, imgsz=640,
                                 tracker="botsort.yaml", persist=True, verbose=False)[0]

            for box in results.boxes:
                if box.id is None: continue
                tid = int(box.id.item())
                conf = box.conf.item()
                g_conf = f"{conf:.3f}"

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                gx1, gy1, gx2, gy2 = rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2

                last_boxes[tid] = (gx1, gy1, gx2, gy2)
                cached_wagons[tid] = frame_count

                if conf > best_conf and conf >= MIN_CONFIDENCE_OCR:
                    best_conf = conf
                    best_box = (gx1, gy1, gx2, gy2)
                    best_id = tid

        # OCR-ისთვის საუკეთესო კადრის გაგზავნა
        if detection_active and best_box and frame_count % 12 == 0:
            x1, y1, x2, y2 = best_box
            crop = frame[y1:y2, x1:x2]
            if crop.size and crop_queue.qsize() < 2:
                crop_queue.put((crop.copy(), best_id))

        # ხატვა
        for tid, (x1, y1, x2, y2) in last_boxes.items():
            if tid not in cached_wagons: continue
            dx1 = int(x1 * scale) + xo
            dy1 = int(y1 * scale) + yo
            dx2 = int(x2 * scale) + xo
            dy2 = int(y2 * scale) + yo

            color = (0, 165, 255)
            if tid in wagon_numbers: color = (0, 255, 0)
            elif tid == best_id: color = (0, 255, 255)

            cv2.rectangle(disp, (dx1, dy1), (dx2, dy2), color, 3)
            label = f"{track_to_display_id.get(tid, '?')}-->{wagon_numbers.get(tid, '?')}"
            cv2.putText(disp, label, (dx1, dy1-10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # === ახალი: დეტექციის სტატუსის ჩვენება GUI + HLS-ზე ===
        status_text = " weighing " if detection_active else ""
        status_color = (0, 255, 0) if detection_active else (0, 0, 255)   # მწვანე / წითელი
        cv2.putText(disp, status_text, (600, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, status_color, 4)

        # დანარჩენი ინფორმაცია
        cv2.putText(disp, f"Conf: {g_conf}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
        cv2.putText(disp, f"Unique: {wagons_data['session']['unique_numbers']}", (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)

        # HLS გაგზავნა
        if ffmpeg_process and ffmpeg_process.poll() is None:
            try:
                ffmpeg_process.stdin.write(disp.tobytes())
                ffmpeg_process.stdin.flush()
            except:
                print("FFmpeg გავიდა — ვრთავ ხელახლა")
                ffmpeg_process = None
                start_hls_stream()

        cv2.imshow("WAGON TRACKER", disp)
        if cv2.waitKey(1) == 27:
            break

    cleanup()

def cleanup():
    global running, ffmpeg_process
    print("\nგაჩერება...")
    running = False
    time.sleep(0.5)
    if cap: cap.release()
    cv2.destroyAllWindows()
    if ffmpeg_process:
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        ffmpeg_process.wait()

if __name__ == "__main__":
    main()