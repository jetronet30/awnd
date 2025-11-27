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

# **სრული GUI კონტროლი + WAGON OCR**
cv2.setNumThreads(1)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF_ENABLE_OPENEXR"] = "0"
os.environ["OPENCV_SHOW_IMAGES"] = "0"

# ================================
# კონფიგურაცია
# ================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hls_output")
HLS_PLAYLIST = os.path.join(OUTPUT_DIR, "index.m3u8")
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"
SEGMENT_DURATION = 5
MODEL_PATH = "best.pt"
LOG_FILE = os.path.join(os.path.dirname(__file__), "wagon_ocr_results.txt")

# **WAGON OCR კონფიგი (ოპტიმიზებული)**
MIN_CONFIDENCE_OCR = 0.83
MATCH_THRESHOLD = 180
MIN_CONFIDENCE_FOR_ID = 0.85

FIXED_WINDOW_WIDTH = 1280
FIXED_WINDOW_HEIGHT = 720

LEFT_MARGIN   = 0.20    
RIGHT_MARGIN  = 0.20    
TOP_MARGIN    = 0.20    
BOTTOM_MARGIN = 0.20    

# **გლობალური ცვლადები (ოპტიმიზებული)**
frame_queue = queue.Queue(maxsize=10)
crop_queue = queue.Queue(maxsize=12)
ffmpeg_process = None
running = True
model = None
cap = None
last_ocr_text = "wagon: -"
ocr_lock = threading.Lock()
known_sectors = {}
next_id = 1

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_stable_id(current_center, confidence):
    global next_id
    if confidence < MIN_CONFIDENCE_FOR_ID:
        return None
    best_id = None
    best_distance = float('inf')
    for sid, known_center in known_sectors.items():
        dist = ((current_center[0] - known_center[0])**2 +
                (current_center[1] - known_center[1])**2)**0.5
        if dist < best_distance and dist < MATCH_THRESHOLD:
            best_distance = dist
            best_id = sid
    if best_id is not None:
        known_sectors[best_id] = current_center
        return best_id
    else:
        new_id = next_id
        known_sectors[new_id] = current_center
        next_id += 1
        return new_id

# **OCR Worker Thread (ოპტიმიზებული სიჩქარისთვის)**
def ocr_worker():
    global last_ocr_text, running
    
    print("[INFO] TrOCR მოდელი იტვირთება...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    trocr_model.to("cpu")
    trocr_model.eval()
    torch.set_grad_enabled(False)
    
    # TXT ლოგის ინიციალიზაცია
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== HLS + WAGON OCR: ახალი სესია " + "="*50 + "\n")
    
    while running:
        try:
            item = crop_queue.get(timeout=0.3)
            if item is None:
                break
            cropped_img, wagon_id = item

            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize((384, 96), Image.BILINEAR)
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated_ids = trocr_model.generate(
                    pixel_values,
                    max_length=12,
                    num_beams=1,
                    early_stopping=True
                )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            cleaned = re.sub(r'[^\d]', '', text
                             .replace('O', '0').replace('o', '0')
                             .replace('I', '1').replace('l', '1')
                             .replace('S', '5').replace('B', '8'))

            if len(cleaned) >= 4:
                result = f"wagon-{wagon_id}: {cleaned}"
                with ocr_lock:
                    last_ocr_text = result
                if running:
                    print(f"[OCR] {result}")

                    with open(LOG_FILE, "a", encoding="utf-8") as logf:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        logf.write(f"[{timestamp}] {result}\n")

        except queue.Empty:
            continue
        except Exception as e:
            if running:
                print(f"[OCR შეცდომა] {e}")

# **OCR თრედის გაშვება**
ocr_thread = threading.Thread(target=ocr_worker, daemon=False)
ocr_thread.start()

def signal_handler(sig, frame):
    global running
    print("\n⏹️ გაჩერდა Ctrl+C-ით...")
    running = False
    sys.exit(0)

def cleanup_hls():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        try:
            if os.path.isfile(fp) and (f.endswith(".ts") or f == "index.m3u8"):
                os.remove(fp)
        except:
            pass

def start_ffmpeg(width, height, fps):
    global ffmpeg_process
    cleanup_hls()
    
    ffmpeg_cmd = [
        "ffmpeg", "-re", "-y",
        "-f", "rawvideo", 
        "-vcodec", "rawvideo", 
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", 
        "-r", str(fps), 
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-tune", "zerolatency",
        "-g", str(int(fps) * 2),
        "-sc_threshold", "0",
        "-f", "hls",
        "-hls_time", str(SEGMENT_DURATION),
        "-hls_list_size", "10",
        "-hls_flags", "delete_segments+append_list+program_date_time+independent_segments",
        "-hls_segment_filename", os.path.join(OUTPUT_DIR, "segment_%03d.ts"),
        HLS_PLAYLIST
    ]
    
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        print("✅ FFmpeg გაშვებულია")
        return True
    except Exception as e:
        print(f"❌ FFmpeg შეცდომა: {e}")
        return False

def rtsp_reader_thread():
    global cap, running
    
    while running:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                
            ret, frame = cap.read()
            if ret and frame_queue.qsize() < 8:
                frame_queue.put(frame, block=False)
            elif not ret:
                if cap:
                    cap.release()
                cap = None
                time.sleep(1)
                
        except:
            if cap:
                cap.release()
            cap = None
            time.sleep(1)

def cleanup():
    global running, ffmpeg_process, cap
    
    running = False
    
    # OCR queue cleanup
    try:
        crop_queue.put_nowait(None)
    except:
        pass
    
    # ყველა cleanup
    for i in range(10):
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        time.sleep(0.01)
    
    if cap:
        cap.release()
        cap = None
        
    if ffmpeg_process:
        try:
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=3)
        except:
            try:
                ffmpeg_process.kill()
            except:
                pass
    
    # OCR თრედის შეჩერება
    ocr_thread.join(timeout=3)
    
    print(f"\n✅ **დასრულდა!**")
    print(f"📺 HLS: {HLS_PLAYLIST.replace(chr(92), '/')}")
    print(f"💾 Wagon ლოგი: {LOG_FILE}")
    print(f"🔢 სულ ინდექსირებული wagon: {next_id-1}")
    sys.exit(0)

def main_loop():
    global running, model, cap, width, height
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # YOLO მოდელი
    try:
        model = YOLO(MODEL_PATH)
        model.overrides['show'] = False
        model.overrides['save'] = False
        model.overrides['visualize'] = False
        print(f"✅ YOLO + OCR ჩაიტვირთა!")
    except Exception as e:
        print(f"❌ YOLO შეცდომა: {e}")
        return
    
    # RTSP ტესტი
    test_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = test_cap.get(cv2.CAP_PROP_FPS) or 25
    test_cap.release()
    
    if width == 0 or height == 0:
        print("❌ კამერა ვერ გაიხსნა!")
        return
    
    if not start_ffmpeg(width, height, fps):
        return
    
    # ROI
    x1 = int(width * LEFT_MARGIN)
    x2 = int(width * (1 - RIGHT_MARGIN))
    y1 = int(height * TOP_MARGIN)
    y2 = int(height * (1 - BOTTOM_MARGIN))
    
    print(f"\n🚂 **WAGON OCR + HLS STREAM (⚡ FPS ოპტიმიზებული)**")
    print(f"📺 რეზოლუცია: {width}x{height}")
    print(f"🎯 TRAIN ზონა: ({x1},{y1},{x2},{y2})")
    print(f"💾 OCR ლოგი: {LOG_FILE}")
    print(f"📡 HLS: {HLS_PLAYLIST.replace(chr(92), '/')}")
    
    # RTSP თრედი
    rtsp_thread = threading.Thread(target=rtsp_reader_thread, daemon=True)
    rtsp_thread.start()
    time.sleep(2)
    
    # ფანჯარა
    window_name = "🚂 WAGON OCR + HLS LIVE ⚡"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FIXED_WINDOW_WIDTH, FIXED_WINDOW_HEIGHT)
    cv2.moveWindow(window_name, 50, 30)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # 🔥 **FPS ოპტიმიზაციის ცვლადები**
    frame_count = 0
    last_fps_time = time.time()
    current_fps = fps
    best_conf_local = 0.0
    best_id_local = 0
    yolo_frame_skip = 0      # YOLO ყოველ 3 ფრეიმზე
    ffmpeg_skip = 0          # FFmpeg ყოველ 2 ფრეიმზე
    ocr_frame_count = 0      # OCR ყოველ 20 ფრეიმზე
    
    # **ძველი detection-ის კეში** (სტაბილურობისთვის)
    cached_boxes = []
    
    print("🎬 **დაიწყო WAGON ნომრის ამოღება!**")
    print("⏹️ **გაჩერება:** 'q' ან Ctrl+C")
    print("⚡ **YOLO: ყოველ 3 ფრეიმზე | FFmpeg: ყოველ 2 ფრეიმზე**")
    
    try:
        while running:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # **YOLO SKIP LOGIC**
            yolo_frame_skip += 1
            do_yolo = (yolo_frame_skip % 3 == 0)
            ffmpeg_skip += 1
            do_ffmpeg = (ffmpeg_skip % 2 == 0)
            ocr_frame_count += 1
            
            # რეზიზი და display_frame
            scale_w = FIXED_WINDOW_WIDTH / width
            scale_h = FIXED_WINDOW_HEIGHT / height
            scale = min(scale_w, scale_h)
            
            new_w = int(width * scale)
            new_h = int(height * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            display_frame = np.zeros((FIXED_WINDOW_HEIGHT, FIXED_WINDOW_WIDTH, 3), dtype=np.uint8)
            x_offset = (FIXED_WINDOW_WIDTH - new_w) // 2
            y_offset = (FIXED_WINDOW_HEIGHT - new_h) // 2
            display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
            
            frame_copy = display_frame.copy()
            
            # ROI (მხოლოდ ჩრდილი)
            roi_x1 = int(x1 * scale) + x_offset
            roi_y1 = int(y1 * scale) + y_offset
            roi_x2 = int(x2 * scale) + x_offset
            roi_y2 = int(y2 * scale) + y_offset
            cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            
            # **WAGON DETECTION მხოლოდ ყოველ 3 ფრეიმზე**
            roi_frame = frame[y1:y2, x1:x2]
            best_sector = None
            best_conf_local = 0.0
            best_id_local = 0
            
            if do_yolo and roi_frame.size > 0 and model:
                try:
                    results = model(roi_frame, verbose=False, conf=0.3, show=False)[0]
                    cached_boxes = []  # ყოველ YOLO-ს შემდეგ განახლება
                    
                    for box in results.boxes:
                        rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        
                        # გლობალური კოორდინატები
                        gx1, gy1 = x1 + rx1, y1 + ry1
                        gx2, gy2 = x1 + rx2, y1 + ry2
                        
                        # ცენტრი და ID
                        center = get_center((gx1, gy1, gx2, gy2))
                        wagon_id = get_stable_id(center, conf)
                        
                        # კეში შენახვა
                        cached_boxes.append((gx1, gy1, gx2, gy2, conf, wagon_id))
                        
                        # საუკეთესო wagon OCR-სთვის
                        if conf > best_conf_local and conf >= MIN_CONFIDENCE_OCR and wagon_id:
                            best_conf_local = conf
                            best_sector = (gx1, gy1, gx2, gy2)
                            best_id_local = wagon_id
                            
                except Exception as e:
                    pass
            
            # **ძველი BOX-ების გამოტანა** (სტაბილურობისთვის)
            for gx1, gy1, gx2, gy2, conf, wagon_id in cached_boxes:
                display_gx1 = int(gx1 * scale) + x_offset
                display_gy1 = int(gy1 * scale) + y_offset
                display_gx2 = int(gx2 * scale) + x_offset
                display_gy2 = int(gy2 * scale) + y_offset
                
                if conf >= MIN_CONFIDENCE_OCR:
                    color = (0, 255, 0)
                    label = f"W{wagon_id}"
                else:
                    color = (0, 120, 255)
                    label = ""
                
                cv2.rectangle(frame_copy, (display_gx1, display_gy1), 
                            (display_gx2, display_gy2), color, 3)
                if label:
                    cv2.putText(frame_copy, label, (display_gx1, display_gy1 - 10),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            
            # **OCR ყოველ 20 ფრეიმზე**
            if best_sector and ocr_frame_count % 20 == 0:
                bx1, by1, bx2, by2 = best_sector
                cropped = frame[by1:by2, bx1:bx2]
                try:
                    if crop_queue.qsize() >= 10:
                        try:
                            crop_queue.get_nowait()
                        except:
                            pass
                    crop_queue.put_nowait((cropped.copy(), best_id_local))
                except queue.Full:
                    pass
            
            # *** GUI ტექსტები ***
            with ocr_lock:
                current_ocr_text = last_ocr_text
            
            frame_count += 1
            if time.time() - last_fps_time > 1.0:
                current_fps = frame_count / (time.time() - last_fps_time)
                last_fps_time = time.time()
                frame_count = 0
            
            # 1. FPS
            cv2.putText(frame_copy, f"FPS: {current_fps:.1f}", 
                       (FIXED_WINDOW_WIDTH - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 2. YOLO სტატუსი
            yolo_status = "ON" if do_yolo else "OFF"
            yolo_color = (0, 255, 0) if do_yolo else (0, 255, 255)
            cv2.putText(frame_copy, f"YOLO: {yolo_status}", 
                       (FIXED_WINDOW_WIDTH - 150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, yolo_color, 2)
            
            # 3. OCR Queue
            cv2.putText(frame_copy, f"OCRQ: {crop_queue.qsize()}/12", 
                       (FIXED_WINDOW_WIDTH - 150, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 4. ვაგონის ნომერი
            cv2.putText(frame_copy, current_ocr_text, 
                       (20, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 3)
            
            # 5. რაოდენობა
            cv2.putText(frame_copy, f"Wagons: {next_id-1}", 
                       (FIXED_WINDOW_WIDTH - 300, FIXED_WINDOW_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 6. Confidence
            conf_text = f"Conf: {best_conf_local:.1f}"
            conf_color = (0, 255, 0) if best_conf_local >= 0.9 else (0, 255, 255) if best_conf_local >= 0.5 else (0, 120, 255)
            cv2.putText(frame_copy, conf_text, 
                       (20, FIXED_WINDOW_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
            
            cv2.imshow(window_name, frame_copy)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF  # 2 → 1 (სწრაფი)
            if key == ord('q'):
                break
            
            # 🔥 **FFmpeg მხოლოდ ყოველ 2 ფრეიმზე**
            if do_ffmpeg and ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                try:
                    orig_frame = frame.copy()
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # **FFmpeg-ში ძველი BOX-ები** (YOLO არ ვაკეთებთ)
                    for gx1, gy1, gx2, gy2, conf, wagon_id in cached_boxes:
                        color = (0, 255, 0) if conf >= MIN_CONFIDENCE_OCR else (0, 120, 255)
                        cv2.rectangle(orig_frame, (gx1, gy1), (gx2, gy2), color, 4)
                        cv2.putText(orig_frame, f"W{conf:.1f}", (gx1, gy1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # OCR text
                    with ocr_lock:
                        cv2.putText(orig_frame, last_ocr_text, (20, 90),
                                   cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 5)
                    
                    cv2.putText(orig_frame, f"FPS: {current_fps:.1f}", (width - 150, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    ffmpeg_process.stdin.write(orig_frame.tobytes())
                    ffmpeg_process.stdin.flush()
                except:
                    pass
                    
    finally:
        cleanup()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
    main_loop()