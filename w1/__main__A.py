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

# ========================================
# **სისტემური ოპტიმიზაციები (საჭიროა სტაბილურობისთვის)**
# ========================================
cv2.setNumThreads(1)  # OpenCV მხოლოდ 1 თრედს იყენებს (სტაბილურობა)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"  # OpenEXR დამოუკიდებლობის გამორთვა
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF_ENABLE_OPENEXR"] = "0"
os.environ["OPENCV_SHOW_IMAGES"] = "0"  # ფანჯრების ავტო-გამოჩენის გამორთვა

# ========================================
# **კონფიგურაციის ფაილები და პარამეტრები**
# ========================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hls_output")  # HLS ვიდეოს საქაღალდე
HLS_PLAYLIST = os.path.join(OUTPUT_DIR, "index.m3u8")  # HLS playlist ფაილი
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"  # **კამერის RTSP მისამართი** (შეცვალე საკუთარით!)
SEGMENT_DURATION = 5  # HLS სეგმენტის ხანგრძლივობა (წამში)
MODEL_PATH = "best.pt"  # **YOLO მოდელის მისამართი** (wagon detection-ისთვის)
LOG_FILE = os.path.join(os.path.dirname(__file__), "wagon_ocr_results.txt")  # OCR შედეგების ლოგი

# ========================================
# **OCR (ტექსტის ამოცნობის) პარამეტრები**
# ========================================
MIN_CONFIDENCE_OCR = 0.6  # მინიმალური სანდოობა OCR-ისთვის
MATCH_THRESHOLD = 180  # მანძილი პიქსელებში (wagon ID შესაბამისლად)
MIN_CONFIDENCE_FOR_ID = 0.6  # მინიმალური სანდოობა wagon ID-ის მისანიჭებლად

# ========================================
# **GUI ფანჯრის ზომები (ფიქსირებული)**
# ========================================
FIXED_WINDOW_WIDTH = 1280   # ფანჯრის სიგანე
FIXED_WINDOW_HEIGHT = 720   # ფანჯრის სიმაღლე

# ========================================
# **ROI (Region of Interest) - TRAIN-ის შეხედვის ზონა**
# ========================================
LEFT_MARGIN   = 0.20    # მარცხენა ზღვარი (20% ვიდეოს სიგანისგან)
RIGHT_MARGIN  = 0.20    # მარჯვენა ზღვარი (20% ვიდეოს სიგანისგან)  
TOP_MARGIN    = 0.20    # ზედა ზღვარი (20% ვიდეოს სიმაღლისგან)
BOTTOM_MARGIN = 0.20    # ქვედა ზღვარი (20% ვიდეოს სიმაღლისგან)

# ========================================
# **გლობალური ცვლადები (მრავალთრედიანი სინქრონიზაციისთვის)**
# ========================================
frame_queue = queue.Queue(maxsize=10)      # ფრეიმების რიგი (RTSP → YOLO)
crop_queue = queue.Queue(maxsize=12)       # OCR-ისთვის მზად კადრების რიგი
ffmpeg_process = None                      # FFmpeg პროცესი (HLS სტრიმინგი)
running = True                             # პროგრამის მუშაობის მდგომარეობა
model = None                               # YOLO მოდელი
cap = None                                 # RTSP კამერის ობიექტი
last_ocr_text = "wagon: -"                 # ბოლო წაკითხული wagon ნომერი
ocr_lock = threading.Lock()                # OCR ტექსტის სინქრონიზაციის ლოკი
known_sectors = {}                         # შეხედული wagon-ების ID-ები
next_id = 1                                # შემდეგი wagon ID

# ========================================
# **1. Wagon-ის ცენტრის გამოთვლა**
# ========================================
def get_center(box):
    """
    მიიღებს: wagon-ის ჩარჩოს [x1, y1, x2, y2]
    აბრუნებს: ჩარჩოს ცენტრს (x, y)
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# ========================================
# **2. სტაბილური Wagon ID-ის მინიჭება**
# ========================================
def get_stable_id(current_center, confidence):
    """
    ლოგიკა:
    1. თუ confidence < MIN_CONFIDENCE_FOR_ID → None
    2. ეძებს ყველაზე ახლომდებარე ცნობილ wagon-ს
    3. თუ მანძილი < MATCH_THRESHOLD → იგივე ID
    4. თუ არა → ახალი ID ქმნის
    """
    global next_id
    if confidence < MIN_CONFIDENCE_FOR_ID:
        return None
    
    best_id = None
    best_distance = float('inf')
    
    # ყველა ცნობილი wagon-ის შემოწმება
    for sid, known_center in known_sectors.items():
        # ევკლიდური მანძილი ორ ცენტრს შორის
        dist = ((current_center[0] - known_center[0])**2 +
                (current_center[1] - known_center[1])**2)**0.5
        
        if dist < best_distance and dist < MATCH_THRESHOLD:
            best_distance = dist
            best_id = sid
    
    # შედეგი
    if best_id is not None:
        # განახლება ძველი ID-ის პოზიცია
        known_sectors[best_id] = current_center
        return best_id
    else:
        # ახალი ID
        new_id = next_id
        known_sectors[new_id] = current_center
        next_id += 1
        return new_id

# ========================================
# **3. OCR Worker Thread (ძირითადი OCR ლოგიკა)**
# ========================================
def ocr_worker():
    """
    მუშაობს ცალკე თრედში:
    1. TrOCR მოდელის ჩატვირთვა
    2. ყოველთვის იღებს crop_queue-დან მზად wagon კადრს
    3. აკეთებს OCR-ს
    4. ინახავს შედეგს last_ocr_text-ში
    5. ყველაფერს ლოგავს ფაილში
    """
    global last_ocr_text, running
    
    print("[INFO] 🎯 TrOCR მოდელი იტვირთება...")
    
    # **TrOCR მოდელის ჩატვირთვა (printed text-ისთვის ოპტიმიზებული)**
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    trocr_model.to("cpu")           # CPU-ზე (GPU არ არის საჭირო)
    trocr_model.eval()              # Inference რეჟიმი
    torch.set_grad_enabled(False)   # Gradient-ის გამოთვლის გამორთვა (სიჩქარე)
    
    # **ლოგ ფაილის ინიციალიზაცია**
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== 🚂 HLS + WAGON OCR: ახალი სესია " + "="*50 + "\n")
        f.write(f"[{datetime.now()}] სტარტი\n")
    
    while running:
        try:
            # **იღებს მზად wagon კადრს რიგიდან**
            item = crop_queue.get(timeout=0.3)
            if item is None:  # შეჩერების სიგნალი
                break
                
            cropped_img, wagon_id = item  # [კადრი, wagon_id]

            # **კადრის მომზადება OCR-ისთვის**
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize((384, 96), Image.BILINEAR)  # ოპტიმალური ზომა
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values

            # **OCR გაშვება**
            with torch.no_grad():  # მეხსიერების ოპტიმიზაცია
                generated_ids = trocr_model.generate(
                    pixel_values,
                    max_length=12,           # მაქს 12 სიმბოლო
                    num_beams=1,             # სწრაფი (არა ნელი ძიება)
                    early_stopping=True      # ადრეული შეჩერება
                )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # **ტექსტის გაწმენდა**
            cleaned = re.sub(r'[^\d]', '', text  # მხოლოდ ციფრები
                            .replace('O', '0').replace('o', '0')    # O → 0
                            .replace('I', '1').replace('l', '1')    # I/l → 1
                            .replace('S', '5').replace('B', '8'))   # S→5, B→8

            # **შედეგის შენახვა თუ საკმარისია**
            if len(cleaned) >= 4:  # მინიმუმ 4 ციფრი
                result = f"wagon-{wagon_id}: {cleaned}"
                
                with ocr_lock:  # თრედის უსაფრთხოება
                    last_ocr_text = result
                
                if running:
                    print(f"[OCR ✅] {result}")

                    # **ლოგ ფაილში ჩაწერა**
                    with open(LOG_FILE, "a", encoding="utf-8") as logf:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        logf.write(f"[{timestamp}] {result}\n")

        except queue.Empty:  # რიგი ცარიელია
            continue
        except Exception as e:
            if running:
                print(f"[OCR ⚠️] შეცდომა: {e}")

# ========================================
# **4. OCR თრედის გაშვება**
# ========================================
print("[START] 🚂 OCR თრედი იწყება...")
ocr_thread = threading.Thread(target=ocr_worker, daemon=False)  # მთავარი თრედი
ocr_thread.start()

# ========================================
# **5. სისტემური სიგნალების მართვა**
# ========================================
def signal_handler(sig, frame):
    """Ctrl+C-ის მართვა"""
    global running
    print("\n⏹️ გაჩერდა Ctrl+C-ით...")
    running = False
    sys.exit(0)

# ========================================
# **6. HLS საქაღალდის გაწმენდა**
# ========================================
def cleanup_hls():
    """ძველი HLS ფაილების წაშლა"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        try:
            if os.path.isfile(fp) and (f.endswith(".ts") or f == "index.m3u8"):
                os.remove(fp)
                print(f"[HLS] 🗑️ წაშლილი: {f}")
        except:
            pass

# ========================================
# **7. FFmpeg HLS სტრიმინგის დაწყება**
# ========================================
def start_ffmpeg(width, height, fps):
    """
    FFmpeg კომანდა:
    - rawvideo → libx264 → HLS
    - zerolatency: რეალურ დროში
    - delete_segments: ძველი სეგმენტების ავტო წაშლა
    """
    global ffmpeg_process
    cleanup_hls()
    
    ffmpeg_cmd = [
        "ffmpeg", "-re", "-y",                           # რეალური დრო, ზებრივი ჩაწერა
        "-f", "rawvideo",                                # შეყვანა: raw BGR24
        "-vcodec", "rawvideo", 
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",                       # რეზოლუცია
        "-r", str(fps),                                  # FPS
        "-i", "-",                                       # stdin
        "-c:v", "libx264",                               # H.264 კოდეკი
        "-preset", "fast",                               # სწრაფი კოდირება
        "-tune", "zerolatency",                          # ნულოვანი დაგვიანება
        "-g", str(int(fps) * 2),                         # GOP ზომა
        "-sc_threshold", "0",                            # scene cut გამორთვა
        "-f", "hls",                                     # HLS ფორმატი
        "-hls_time", str(SEGMENT_DURATION),              # სეგმენტის ხანგრძლივობა
        "-hls_list_size", "10",                          # მაქს 10 სეგმენტი
        "-hls_flags", "delete_segments+append_list+program_date_time+independent_segments",
        "-hls_segment_filename", os.path.join(OUTPUT_DIR, "segment_%03d.ts"),
        HLS_PLAYLIST                                     # playlist
    ]
    
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,                                   # დიდი ბუფერი
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # Windows-ისთვის
        )
        return True
    except Exception as e:
        print(f"❌ FFmpeg შეცდომა: {e}")
        return False

# ========================================
# **8. RTSP კამერის წაკითხვის თრედი**
# ========================================
def rtsp_reader_thread():
    """ცალკე თრედი RTSP-დან ფრეიმების წამოსაკითხად"""
    global cap, running
    
    while running:
        try:
            # **კამერის რეკონექტი თუ დაკარგა**
            if cap is None or not cap.isOpened():
                print("[RTSP] 🔄 რეკონექტი...")
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # მხოლოდ 1 ფრეიმის ბუფერი
                cap.set(cv2.CAP_PROP_FPS, 25)
                
            ret, frame = cap.read()
            if ret and frame_queue.qsize() < 8:  # რიგი არ სავსეა
                frame_queue.put(frame, block=False)
            elif not ret:  # კამერა დაკარგა
                if cap:
                    cap.release()
                cap = None
                time.sleep(1)  # 1 წამი ლოდინი
                
        except Exception as e:
            print(f"[RTSP ⚠️] {e}")
            if cap:
                cap.release()
            cap = None
            time.sleep(1)

# ========================================
# **9. სრული გაწმენდა და შეჩერება**
# ========================================
def cleanup():
    """ყველაფრის უსაფრთხო გაწმენდა"""
    global running, ffmpeg_process, cap
    
    print("\n🧹 **გაწმენდა იწყება...**")
    running = False
    
    # **OCR რიგის შეჩერება**
    try:
        crop_queue.put_nowait(None)
    except:
        pass
    
    # **OpenCV ფანჯრების დახურვა**
    for i in range(10):
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        time.sleep(0.01)
    
    # **კამერის გაქცევა**
    if cap:
        cap.release()
        cap = None
        
    # **FFmpeg შეჩერება**
    if ffmpeg_process:
        try:
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=3)
            print("✅ FFmpeg შეჩერდა")
        except:
            try:
                ffmpeg_process.kill()
                print("⚡ FFmpeg ძალით შეჩერდა")
            except:
                pass
    
    # **OCR თრედის ლოდინი**
    try:
        ocr_thread.join(timeout=3)
        print("✅ OCR თრედი შეჩერდა")
    except:
        pass
    
    # **ფინალური ინფორმაცია**
    print(f"\n🎉 **დასრულდა წარმატებით!**")
    print(f"📺 HLS სტრიმი: {HLS_PLAYLIST.replace(chr(92), '/')}")
    print(f"💾 Wagon ლოგი: {LOG_FILE}")
    print(f"🔢 სულ ინდექსირებული wagon-ები: {next_id-1}")
    sys.exit(0)

# ========================================
# **10. მთავარი ციკლი (ძირითადი ლოგიკა)**
# ========================================
def main_loop():
    global running, model, cap, width, height
    
    # **Ctrl+C მართვა**
    signal.signal(signal.SIGINT, signal_handler)
    
    # ========================================
    # **YOLO მოდელის ჩატვირთვა**
    # ========================================
    try:
        print(f"[YOLO] 🚂 მოდელი იტვირთება: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.overrides['show'] = False      # ვიზუალიზაციის გამორთვა
        model.overrides['save'] = False      # შენახვის გამორთვა
        model.overrides['visualize'] = False # ნერვული ქსელის ვიზუალიზაციის გამორთვა
        print(f"✅ YOLOv8 + TrOCR ოპტიმიზებული სისტემა ჩაიტვირთა!")
    except Exception as e:
        print(f"❌ YOLO შეცდომა: {e}")
        print("💡 შეამოწმე MODEL_PATH და ultralytics დაყენება!")
        return
    
    # ========================================
    # **კამერის ტესტირება**
    # ========================================
    print("[RTSP] 📷 კამერის ტესტირება...")
    test_cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = test_cap.get(cv2.CAP_PROP_FPS) or 25
    test_cap.release()
    
    if width == 0 or height == 0:
        print("❌ 🚫 კამერა ვერ გაიხსნა!")
        print(f"💡 შეამოწმე RTSP_URL: {RTSP_URL}")
        return
    
    print(f"✅ 📷 კამერა: {width}x{height} @ {fps}fps")
    
    # ========================================
    # **FFmpeg HLS სტრიმინგის დაწყება**
    # ========================================
    if not start_ffmpeg(width, height, fps):
        return
    
    # ========================================
    # **ROI ზონის გამოთვლა**
    # ========================================
    x1 = int(width * LEFT_MARGIN)
    x2 = int(width * (1 - RIGHT_MARGIN))
    y1 = int(height * TOP_MARGIN)
    y2 = int(height * (1 - BOTTOM_MARGIN))
    
    # **საწყისი ინფორმაცია**
    print(f"\n🚂 **=== WAGON OCR + LIVE HLS STREAMING ===**")
    print(f"📺 რეზოლუცია: {width}x{height} @ {fps}fps")
    print(f"🎯 **TRAIN ROI ზონა**: ({x1},{y1}) → ({x2},{y2})")
    print(f"💾 OCR ლოგი: {LOG_FILE}")
    print(f"📡 HLS სტრიმი: {HLS_PLAYLIST.replace(chr(92), '/')}")
    print(f"⚙️  OCR Confidence: ≥{MIN_CONFIDENCE_OCR}")
    
    # ========================================
    # **RTSP თრედის გაშვება**
    # ========================================
    rtsp_thread = threading.Thread(target=rtsp_reader_thread, daemon=True)
    rtsp_thread.start()
    time.sleep(2)  # ლოდინი სტაბილიზაციისთვის
    
    # ========================================
    # **GUI ფანჯრის შექმნა**
    # ========================================
    window_name = "🚂 WAGON OCR + LIVE HLS STREAM ⚡"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FIXED_WINDOW_WIDTH, FIXED_WINDOW_HEIGHT)
    cv2.moveWindow(window_name, 50, 30)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # ზევით ყოველთვის
    
    # ========================================
    # **FPS ოპტიმიზაციის ცვლადები**
    # ========================================
    frame_count = 0
    last_fps_time = time.time()
    current_fps = fps
    
    # **ოპტიმიზაციის კონტროლერები**
    yolo_frame_skip = 0      # YOLO ყოველ 3 ფრეიმზე (სიჩქარე ↑)
    ffmpeg_skip = 0          # FFmpeg ყოველ 2 ფრეიმზე (სტაბილურობა)
    ocr_frame_count = 0      # OCR ყოველ 20 ფრეიმზე (სისტემის დატვირთვა ↓)
    
    # **კეში ძველი detection-ებისთვის (სტაბილურობა)**
    cached_boxes = []
    
    best_conf_local = 0.0
    best_id_local = 0
    
    print("🎬 **=== LIVE WAGON ნომრის ამოღება დაიწყო! ===**")
    print("⏹️  **გაჩერება**: 'q' ღილაკი ან Ctrl+C")
    print("⚡  **ოპტიმიზაცია**: YOLO=3fps | FFmpeg=2fps | OCR=20fps")
    print("-" * 70)
    
    # ========================================
    # **ძირითადი ციკლი**
    # ========================================
    try:
        while running:
            # **ფრეიმის მიღება რიგიდან**
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # **ოპტიმიზაციის ციკლები**
            yolo_frame_skip += 1
            do_yolo = (yolo_frame_skip % 3 == 0)        # YOLO ყოველ 3-ში
            ffmpeg_skip += 1
            do_ffmpeg = (ffmpeg_skip % 2 == 0)           # FFmpeg ყოველ 2-ში
            ocr_frame_count += 1
            
            # ========================================
            # **ფანჯრის რეზიზი და მომზადება**
            # ========================================
            scale_w = FIXED_WINDOW_WIDTH / width
            scale_h = FIXED_WINDOW_HEIGHT / height
            scale = min(scale_w, scale_h)
            
            new_w = int(width * scale)
            new_h = int(height * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # **შავი ფონი + ცენტრირება**
            display_frame = np.zeros((FIXED_WINDOW_HEIGHT, FIXED_WINDOW_WIDTH, 3), dtype=np.uint8)
            x_offset = (FIXED_WINDOW_WIDTH - new_w) // 2
            y_offset = (FIXED_WINDOW_HEIGHT - new_h) // 2
            display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
            
            frame_copy = display_frame.copy()
            
            # **ROI ჩარჩო (ნარინჯისფერი)**
            roi_x1 = int(x1 * scale) + x_offset
            roi_y1 = int(y1 * scale) + y_offset
            roi_x2 = int(x2 * scale) + x_offset
            roi_y2 = int(y2 * scale) + y_offset
            cv2.rectangle(frame_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            
            # ========================================
            # **YOLO Wagon Detection (მხოლოდ ყოველ 3 ფრეიმზე)**
            # ========================================
            roi_frame = frame[y1:y2, x1:x2]  # მხოლოდ ROI ზონა
            best_sector = None
            best_conf_local = 0.0
            best_id_local = 0
            
            if do_yolo and roi_frame.size > 0 and model:
                try:
                    # **YOLO inference**
                    results = model(roi_frame, verbose=False, conf=0.3, show=False)[0]
                    cached_boxes = []  # ყოველ YOLO შემდეგ განახლება
                    
                    for box in results.boxes:
                        # **ბოქსის კოორდინატები**
                        rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        
                        # **გლობალური კოორდინატები (ROI-დან მთელ ფრეიმზე)**
                        gx1, gy1 = x1 + rx1, y1 + ry1
                        gx2, gy2 = x1 + rx2, y1 + ry2
                        
                        # **ცენტრი და სტაბილური ID**
                        center = get_center((gx1, gy1, gx2, gy2))
                        wagon_id = get_stable_id(center, conf)
                        
                        # **კეშში შენახვა**
                        cached_boxes.append((gx1, gy1, gx2, gy2, conf, wagon_id))
                        
                        # **საუკეთესო wagon OCR-ისთვის**
                        if conf > best_conf_local and conf >= MIN_CONFIDENCE_OCR and wagon_id:
                            best_conf_local = conf
                            best_sector = (gx1, gy1, gx2, gy2)
                            best_id_local = wagon_id
                            
                except Exception as e:
                    print(f"[YOLO ⚠️] {e}")
            
            # ========================================
            # **ძველი BOX-ების გამოტანა (სტაბილურობა)**
            # ========================================
            for gx1, gy1, gx2, gy2, conf, wagon_id in cached_boxes:
                # **დისპლეი კოორდინატები**
                display_gx1 = int(gx1 * scale) + x_offset
                display_gy1 = int(gy1 * scale) + y_offset
                display_gx2 = int(gx2 * scale) + x_offset
                display_gy2 = int(gy2 * scale) + y_offset
                
                # **ფერი და ეტიკეტი**
                if conf >= MIN_CONFIDENCE_OCR:
                    color = (0, 255, 0)      # მწვანე - კარგი
                    label = f"W{wagon_id}"
                else:
                    color = (0, 120, 255)    # ნარინჯისფერი - სუსტი
                    label = ""
                
                # **ჩარჩოს და ეტიკეტის გამოტანა**
                cv2.rectangle(frame_copy, (display_gx1, display_gy1), 
                            (display_gx2, display_gy2), color, 3)
                if label:
                    cv2.putText(frame_copy, label, (display_gx1, display_gy1 - 10),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            
            # ========================================
            # **OCR გაშვება (ყოველ 20 ფრეიმზე)**
            # ========================================
            if best_sector and ocr_frame_count % 20 == 0:
                bx1, by1, bx2, by2 = best_sector
                cropped = frame[by1:by2, bx1:bx2]  # wagon-ის კადრის გამოჭრა
                
                try:
                    # **რიგის მართვა (არ გადაიტვირთოს)**
                    if crop_queue.qsize() >= 10:
                        try:
                            crop_queue.get_nowait()  # ძველი გამოდევნა
                        except:
                            pass
                    crop_queue.put_nowait((cropped.copy(), best_id_local))
                except queue.Full:
                    pass  # რიგი სავსეა
            
            # ========================================
            # **GUI ინფორმაციის გამოტანა**
            # ========================================
            # მიმდინარე OCR ტექსტი
            with ocr_lock:
                current_ocr_text = last_ocr_text
            
            # **FPS გამოთვლა**
            frame_count += 1
            if time.time() - last_fps_time > 1.0:
                current_fps = frame_count / (time.time() - last_fps_time)
                last_fps_time = time.time()
                frame_count = 0
            
            # **1. FPS**
            cv2.putText(frame_copy, f"FPS: {current_fps:.1f}", 
                       (FIXED_WINDOW_WIDTH - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # **2. YOLO სტატუსი**
            yolo_status = "ON" if do_yolo else "OFF"
            yolo_color = (0, 255, 0) if do_yolo else (0, 255, 255)
            cv2.putText(frame_copy, f"YOLO: {yolo_status}", 
                       (FIXED_WINDOW_WIDTH - 150, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, yolo_color, 2)
            
            # **3. OCR რიგის ზომა**
            cv2.putText(frame_copy, f"OCRQ: {crop_queue.qsize()}/12", 
                       (FIXED_WINDOW_WIDTH - 150, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # **4. Wagon ნომერი (მთავარი)**
            cv2.putText(frame_copy, current_ocr_text, 
                       (20, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 3)
            
            # **5. სულ wagon-ები**
            cv2.putText(frame_copy, f"Wagons: {next_id-1}", 
                       (FIXED_WINDOW_WIDTH - 300, FIXED_WINDOW_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # **6. Confidence**
            conf_text = f"Conf: {best_conf_local:.1f}"
            conf_color = (0, 255, 0) if best_conf_local >= MIN_CONFIDENCE_OCR else (0, 255, 255)
            cv2.putText(frame_copy, conf_text, 
                       (20, FIXED_WINDOW_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
            
            # **ფანჯრის გამოჩენა**
            cv2.imshow(window_name, frame_copy)
            
            # **'q' ღილაკის შემოწმება**
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # ========================================
            # **FFmpeg-ში ჩაწერა (ყოველ 2 ფრეიმზე)**
            # ========================================
            if do_ffmpeg and ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                try:
                    orig_frame = frame.copy()
                    
                    # **ROI ჩარჩო HLS-ში**
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # **ყველა wagon box HLS-ში**
                    for gx1, gy1, gx2, gy2, conf, wagon_id in cached_boxes:
                        color = (0, 255, 0) if conf >= MIN_CONFIDENCE_OCR else (0, 120, 255)
                        cv2.rectangle(orig_frame, (gx1, gy1), (gx2, gy2), color, 4)
                        cv2.putText(orig_frame, f"W{conf:.1f}", (gx1, gy1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # **OCR ტექსტი HLS-ში**
                    with ocr_lock:
                        cv2.putText(orig_frame, last_ocr_text, (20, 90),
                                   cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 255), 5)
                    
                    # **FPS HLS-ში**
                    cv2.putText(orig_frame, f"FPS: {current_fps:.1f}", (width - 150, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # **ჩაწერა FFmpeg-ში**
                    ffmpeg_process.stdin.write(orig_frame.tobytes())
                    ffmpeg_process.stdin.flush()
                    
                except Exception as e:
                    print(f"[FFmpeg ⚠️] {e}")
                    
    finally:
        cleanup()

# ========================================
# **პროგრამის გაშვება**
# ========================================
if __name__ == "__main__":
    # **საბოლოო Ctrl+C მართვა**
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
    
    print("🚂 **WAGON AUTOMATIC NUMBER RECOGNITION SYSTEM**")
    print("👨‍💻 მიერ: AI Vision Engineer")
    print("=" * 70)
    main_loop()