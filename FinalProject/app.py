import cv2
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import csv
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# --- 1. โหลดโมเดล YOLO (4 คลาส) ---
model_path = 'runs/classify/emotion_weights/yolov8_fer_4classes/weights/best.pt'
model = YOLO(model_path)

# --- 2. ใช้ Haar Cascade สำหรับตรวจจับใบหน้า ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 3. ใช้ไฟล์ CSV ไฟล์เดียวจบ (ใช้ utf-8-sig เพื่อรองรับภาษาไทยใน Excel) ---
LOG_FILE = 'seminar_satisfaction_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Topic', 'Total_Audience', 'Happy', 'Neutral', 'Sad', 'Angry', 'Satisfaction_Rate_%'])

# ตัวแปรกลางเก็บค่าสถานะล่าสุดสำหรับปุ่มกด Mark
current_metrics = {
    "total": 0,
    "stats": {'happy': 0, 'neutral': 0, 'sad': 0, 'angry': 0},
    "rate": 0.0
}

def gen_frames():
    global current_metrics
    cap = cv2.VideoCapture(1) # กล้องเว็บแคมภายนอก (ถ้าไม่ขึ้นเปลี่ยนเป็น 0)
    
    frame_counter = 0
    process_every_n_frames = 15 
    
    global_emotion_stats = {'happy': 0, 'neutral': 0, 'sad': 0, 'angry': 0}
    global_satisfaction = 0.0
    last_known_emotions = [] 

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_counter += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_boxes = []
        for (x, y, w, h) in faces:
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            current_boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        total_people = len(current_boxes)

        if frame_counter % process_every_n_frames == 0 and total_people > 0:
            face_crops = [frame[y:y+h_box, x:x+w_box] for (x, y, w_box, h_box) in current_boxes if h_box > 0 and w_box > 0]
            
            if len(face_crops) > 0:
                yolo_results = model(face_crops, verbose=False) 
                last_known_emotions = [res.names[res.probs.top1].lower() for res in yolo_results]
                
                global_emotion_stats = {'happy': 0, 'neutral': 0, 'sad': 0, 'angry': 0}
                for em in last_known_emotions:
                    if em in global_emotion_stats:
                        global_emotion_stats[em] += 1
                
                global_satisfaction = (global_emotion_stats['happy'] / total_people) * 100 if total_people > 0 else 0.0
                
                # อัปเดตตัวแปรกลาง
                current_metrics["total"] = total_people
                current_metrics["stats"] = global_emotion_stats
                current_metrics["rate"] = global_satisfaction

                # บันทึก Log ปกติอัตโนมัติ (ใช้ utf-8-sig)
                with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "", total_people, global_emotion_stats['happy'], global_emotion_stats['neutral'], global_emotion_stats['sad'], global_emotion_stats['angry'], f"{global_satisfaction:.2f}"])

        for i, (x, y, w_box, h_box) in enumerate(current_boxes):
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            text = last_known_emotions[i].upper() if i < len(last_known_emotions) else "TRACKING"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(frame, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Audience: {total_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Happy: {global_emotion_stats['happy']}  Neutral: {global_emotion_stats['neutral']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Sad: {global_emotion_stats['sad']}    Angry: {global_emotion_stats['angry']}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        color_satisfaction = (0, 255, 0) if global_satisfaction >= 50 else (0, 165, 255)
        cv2.putText(frame, f"Satisfaction Rate: {global_satisfaction:.1f}%", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_satisfaction, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Route สำหรับกดปุ่ม Mark (ใช้ utf-8-sig) ---
@app.route('/mark_timestamp', methods=['POST'])
def mark_timestamp():
    data = request.json
    topic_name = data.get('topic', 'หัวข้อไม่ระบุ')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    total = current_metrics["total"]
    stats = current_metrics["stats"]
    rate = current_metrics["rate"]
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, f"📌 [MARK] {topic_name}", total, stats['happy'], stats['neutral'], stats['sad'], stats['angry'], f"{rate:.2f}"])
        
    return jsonify({
        "status": "success", 
        "message": f"บันทึกหัวข้อ '{topic_name}' ลงไฟล์สำเร็จ!",
        "data": {"timestamp": timestamp, "topic": topic_name, "rate": rate}
    })

# --- API Route สำหรับกดปุ่มสรุปรายงานเป็นไฟล์ TXT (คำนวณเฉลี่ย 30 วิ) ---
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        # อ่านข้อมูลจากไฟล์ CSV
        with open(LOG_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return jsonify({"status": "error", "message": "ยังไม่มีข้อมูลในระบบ กรุณารันกล้องสักพักก่อนครับ"})

        total_rates = []
        marks_summary = []
        
        # เตรียมแปลงเวลา Timestamp ของทุกแถวเป็น datetime object เพื่อให้คำนวณง่ายขึ้น
        for row in rows:
            row['dt'] = datetime.strptime(row['Timestamp'], '%Y-%m-%d %H:%M:%S')
            row['rate_float'] = float(row['Satisfaction_Rate_%'])
            total_rates.append(row['rate_float'])

        avg_rate = sum(total_rates) / len(total_rates) if total_rates else 0
        
        # วนลูปหาแถวที่โดน Mark และคำนวณค่าเฉลี่ย 30 วินาทีหลังจากนั้น
        for i, row in enumerate(rows):
            if "📌 [MARK]" in row['Topic']:
                topic_name = row['Topic'].replace('📌 [MARK] ', '')
                mark_time = row['dt']
                end_time = mark_time + timedelta(seconds=30) # บวกเพิ่มไป 30 วินาที
                
                # กวาดหาข้อมูลอารมณ์ในช่วง 30 วินาทีนี้
                rates_in_window = []
                for j in range(i, len(rows)):
                    if mark_time <= rows[j]['dt'] <= end_time:
                        rates_in_window.append(rows[j]['rate_float'])
                    elif rows[j]['dt'] > end_time:
                        break # ถ้าเลย 30 วินาทีไปแล้ว ให้หยุดลูปค้นหาเพื่อประหยัดเวลา
                
                # คำนวณค่าเฉลี่ยในช่วง 30 วินาที
                if rates_in_window:
                    avg_window_rate = sum(rates_in_window) / len(rates_in_window)
                else:
                    avg_window_rate = row['rate_float'] # กรณีเพิ่งกดไปไม่ถึง 30 วิ แล้วรีบกดรายงานเลย
                    
                marks_summary.append(f"- เวลา {row['Timestamp']} | หัวข้อ: {topic_name} | ความพึงพอใจ (เฉลี่ย 30วิ): {avg_window_rate:.2f}%")

        # สร้างไฟล์ Text สรุปรายงาน
        report_file = 'seminar_summary_report.txt'
        with open(report_file, mode='w', encoding='utf-8-sig') as f:
            f.write("=" * 65 + "\n")
            f.write("           สรุปรายงานความพึงพอใจงานสัมมนา (AI Monitoring)\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"เวลาที่พิมพ์รายงาน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ระยะเวลาที่มีการบันทึกทั้งหมด: {len(rows)} ครั้ง\n")
            f.write(f"ความพึงพอใจเฉลี่ยตลอดงาน: {avg_rate:.2f}%\n\n")
            f.write("-" * 65 + "\n")
            f.write("        สรุปความพึงพอใจแยกตามหัวข้อ (หน่วงเวลาประมวลผล 30 วินาที)\n")
            f.write("-" * 65 + "\n")
            
            if marks_summary:
                for mark in marks_summary:
                    f.write(mark + "\n")
            else:
                f.write("ไม่มีการบันทึกแยกหัวข้อ (ไม่ได้กดปุ่ม Mark Timestamp)\n")
                
            f.write("\n" + "=" * 65 + "\n")

        return jsonify({"status": "success", "message": "สร้างไฟล์ seminar_summary_report.txt สำเร็จ!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)