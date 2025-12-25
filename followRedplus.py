import cv2 as cv
import numpy as np
from collections import deque

# ================= VIDEO RECORD (LinkedIn) =================
record = True          # False yaparsan kayıt olmaz
fps = 30
video_seconds = 10
frame_limit = fps * video_seconds
frame_count = 0

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(
    "linkedin_demo.mp4",
    fourcc,
    fps,
    (640, 480)
)
# ==========================================================

# Color Range
lower_red = np.array([0, 150, 120])
upper_red = np.array([10, 255, 255])

# Smoothing
alpha = 0.6
smooth_cx, smooth_cy = None, None
smooth_area = None

# Lost Tracking
lost_frames = 0
MAX_LOST = 5

# Motion Tracking
prev_cx, prev_cy = None, None
speed = 0
MAX_SPEED = 30
direction = ""

# Distance Tracking
prev_area = None
distance_state = ""

# Motion State
motion_state = ""

#Camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

#Path
points = deque(maxlen=30)
kernel = np.ones((5, 5), np.uint8)

direction_buffer = deque(maxlen = 5)
stable_direction = "None"


while True:
    ret, frame = cap.read()
    if not ret:
        break

#UI Overlay
    overlay = frame.copy()
    cv.rectangle(overlay, (5, 5), (260, 230), (0, 0, 0), -1)
    frame = cv.addWeighted(overlay, 0.4, frame, 0.6, 0)

    h, w = frame.shape[:2]


    hud_speed = "None"
    hud_dir = "None"
    hud_motion = "None"
    hud_dist = "None"


    is_jump = False

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red, upper_red)

# Noise Removal
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv.contourArea)
        area = cv.contourArea(c)

        if cv.contourArea(c) > 200:
            lost_frames = 0

            # Area Smoothing
            if smooth_area is None:
                smooth_area = area
            else:
                smooth_area = alpha * smooth_area + (1 - alpha) * area

            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = box.astype(int)

            #Center Smoothing
            cx, cy = int(rect[0][0]), int(rect[0][1])
            if smooth_cx is None:
                smooth_cx, smooth_cy = cx, cy
            else:
                smooth_cx = int(alpha * smooth_cx + (1 - alpha) * cx)
                smooth_cy = int(alpha * smooth_cy + (1 - alpha) * cy)

            points.appendleft((smooth_cx, smooth_cy))

# Speed Calculation
            if prev_cx is not None:
                dx = smooth_cx - prev_cx
                dy = smooth_cy - prev_cy
                speed = int(np.sqrt(dx*dx + dy*dy))
                
                if speed > MAX_SPEED:
                    is_jump = True
# Direction
                if abs(dx) > abs(dy):
                    raw_direction = "Right" if dx > 0 else "Left"
                else:
                    raw_direction = "Down" if dy > 0 else "Up"

                direction_buffer.append(raw_direction)

                if direction_buffer.count(raw_direction) >= 3:
                    stable_direction = raw_direction


            else:
                speed = 0


            # Prediction
            if prev_cx is not None:
                dx = smooth_cx - prev_cx
                dy = smooth_cy - prev_cy

                predict_scale = 2
                pred_x = int(smooth_cx + dx * predict_scale)
                pred_y = int(smooth_cy + dy * predict_scale)

                cv.circle(frame, (pred_x, pred_y), 6, (0,255,255), -1)
                cv.line(frame,
                        (smooth_cx, smooth_cy),
                        (pred_x, pred_y),
                        (0,255,255),
                        2)
                
            prev_cx, prev_cy = smooth_cx, smooth_cy


# Distance State
            if prev_area is not None:
                if smooth_area > prev_area + 50:
                    distance_state = "Approaching"
                elif smooth_area < prev_area + 50:
                    distance_state = "Moving Away"
                else:
                    distance_state = "Stable"
            else:
                distance_state = ""

            prev_area = smooth_area
# Motion State
            if speed < 3:
                motion_state = "Stopped"
            elif speed > 15:
                motion_state = "Fast"
            else:
                motion_state = "Normal"

# HUD Text

            hud_speed = str(speed)
            hud_dir = stable_direction
            hud_motion = motion_state
            hud_dist = distance_state

            cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv.circle(frame, (smooth_cx, smooth_cy), 5, (0, 0, 255), -1)

        else: 
            lost_frames +=1  
    else:
        lost_frames += 1

    cv.putText(frame, "TRACKING STATUS", (15, 25),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv.putText(frame, "Object : Red", (15, 55),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 2)

    cv.putText(frame, f"Speed  : {hud_speed}", (15, 85),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)

    cv.putText(frame, f"Dir    : {hud_dir}", (15, 115),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)

    cv.putText(frame, f"Motion : {hud_motion}", (15, 145),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv.putText(frame, f"Dist   : {hud_dist}", (15, 175),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
    
    if is_jump:
                cv.putText(frame, "JUMP FILTERED",
                    (15, 205),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 68, 68), 2)
# Motion Arrow
    if len(points) >= 2 and lost_frames == 0 and not is_jump:
        x0, y0 = points[0]
        x1, y1 = points[1]

        dx = x0 - x1
        dy = y0 - y1

        arrow_scale = 3
        end_point = (int(x0 + dx * arrow_scale),int(y0 + dy * arrow_scale))

        cv.arrowedLine(
            frame,
            (x0, y0),
            end_point,
            (255,0,0),
            3,
            tipLength = 0.3
        )
    
#Path Lines
    for i in range(1, len(points)):
        thickness = int(np.sqrt(30 / float(i + 1)) * 2)
        cv.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)

#Reset Tracking
    if lost_frames >= MAX_LOST:
        points.clear()
        smooth_cx, smooth_cy = None, None
        smooth_area = None
        prev_cx, prev_cy = None, None
        speed = 0
        direction = ""
        motion_state = ""
        distance_state = ""

        panel_w, panel_h = 320, 80
        px = w // 2 - panel_w // 2
        py = h // 2 - panel_h // 2

        cv.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        cv.putText(frame, "OBJECT LOST",
                   (px + 40, py + 52),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (255, 68, 68), 3)

    cv.imshow("Advanced Tracking", frame)

    # ================= VIDEO RECORD =================
    if  record and frame_count < frame_limit:
        out.write(frame)
        frame_count += 1
    elif record and frame_count >= frame_limit:
        record = False
        out.release()
        print("🎬 Video recording finished")
# ================================================

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if out.isOpened():
    out.release()