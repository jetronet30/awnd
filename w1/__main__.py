import cv2
import subprocess
import os
from datetime import datetime
from ultralytics import YOLO

# ================================
# კონფიგურაცია
# ================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hls_output")
HLS_PLAYLIST = os.path.join(OUTPUT_DIR, "index.m3u8")
RTSP_URL = "rtsp://admin:admin@192.168.1.11:554"
SEGMENT_DURATION = 5
MODEL_PATH = "yolo11n.pt"  # ან yolo11n.pt თუ უფრო სწრაფი გინდა

# ================================
# ფიქსირებული ROI – შეცვალე შენი კამერის მიხედვით
# ================================
LEFT_MARGIN   = 0.20    
RIGHT_MARGIN  = 0.20    
TOP_MARGIN    = 0.20    
BOTTOM_MARGIN = 0.20    

# ================================
# დანარჩენი კოდი
# ================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
for f in os.listdir(OUTPUT_DIR):
    fp = os.path.join(OUTPUT_DIR, f)
    try:
        if os.path.isfile(fp) and (f.endswith(".ts") or f == "index.m3u8"):
            os.remove(fp)
    except:
        pass

model = YOLO(MODEL_PATH)
print(f"YOLO მოდელი ჩაიტვირთა: {MODEL_PATH}")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("შეცდომა: RTSP ვერ იხსნება!")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# ROI გამოთვლა
x1 = int(width * LEFT_MARGIN)
x2 = int(width * (1 - RIGHT_MARGIN))
y1 = int(height * TOP_MARGIN)
y2 = int(height * (1 - BOTTOM_MARGIN))
ROI = (x1, y1, x2, y2)

print(f"\nრეზოლუცია: {width}x{height} | FPS: {fps:.1f}")
print(f"ძებნის ზონა: {ROI}")
print("YOLO ეძებს მხოლოდ TRAIN-ს (მატარებელს) — მაქსიმალური სიჩქარე!\n")

# FFmpeg – უსასრულო HLS
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-g", str(int(fps) * SEGMENT_DURATION),
    "-f", "hls",
    "-hls_time", str(SEGMENT_DURATION),
    "-hls_list_size", "10",
    "-hls_flags", "delete_segments+append_list+program_date_time+independent_segments",
    "-hls_segment_filename", os.path.join(OUTPUT_DIR, "stream_%Y-%m-%d_%H-%M-%S.ts"),
    "-strftime", "1",
    HLS_PLAYLIST
]

process = subprocess.Popen(
    ffmpeg_cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
)

cv2.namedWindow("YOLO11 - TRAIN DETECTION ONLY", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO11 - TRAIN DETECTION ONLY", 1400, 800)

print("დაიწყო! ეძებს მხოლოდ მატარებელს (train)")
print(f"HLS: {HLS_PLAYLIST.replace(chr(92), '/')}\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ფრეიმი აღარ მოდის... ხელახლა ვცდი...")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        frame_copy = frame.copy()

        # წითელი ჩარჩო ROI-სთვის
        x1, y1, x2, y2 = ROI
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv2.putText(frame_copy, "TRAIN DETECTION ZONE", (x1 + 20, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
        cv2.putText(frame_copy, "TRAIN DETECTION ZONE", (x1 + 20, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        # YOLO მხოლოდ ROI-ში + მხოლოდ კლასი 6 (train)
        roi_frame = frame[y1:y2, x1:x2]
        if roi_frame.size > 0:
            # აი აქაა მთავარი ხრიკი — classes=6 → მხოლოდ train!
            results = model(roi_frame, verbose=False, classes=6)  # ← +70% სიჩქარე!
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()

                    # გადავიტანოთ მთავარ კადრზე
                    gx1, gy1 = x1 + rx1, y1 + ry1
                    gx2, gy2 = x1 + rx2, y1 + ry2

                    # ლამაზი ყვითელი ყუთი + ქართული წარწერა
                    cv2.rectangle(frame_copy, (gx1, gy1), (gx2, gy2), (0, 255, 255), 4)
                    cv2.putText(frame_copy, f"train{conf:.2f}", (gx1, gy1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                    cv2.putText(frame_copy, f"train {conf:.2f}", (gx1, gy1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # LIVE + FPS
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame_copy, f"LIVE - {current_time}", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
        cv2.putText(frame_copy, f"LIVE - {current_time}", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)
        cv2.putText(frame_copy, f"FPS: {fps:.1f}", (width - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

        # ჩვენება + HLS
        cv2.imshow("YOLO11 - TRAIN DETECTION ONLY", frame_copy)
        process.stdin.write(frame_copy.tobytes())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nგაჩერდა მომხმარებლის მიერ")
finally:
    cap.release()
    if process.stdin:
        process.stdin.close()
    process.terminate()
    try:
        process.wait(timeout=5)
    except:
        process.kill()
    cv2.destroyAllWindows()
    print(f"\nდასრულდა! HLS მზადაა:")
    print(f"   → {HLS_PLAYLIST}")
    print("   გახსენი VLC-ში ან ბრაუზერში!")