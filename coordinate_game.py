import cv2
import time
import math
from hand_tracker import HandTracker

tracker = HandTracker("hand_landmarker.task")
cap = cv2.VideoCapture(0)

# 🎯 Target points
target_points = [(2,3), (-1,4), (3,-2)]

user_points = []
clicked = False
click_time = 0

# 🎯 SETTINGS (Adjust these)
TOLERANCE = 0.8      # 👈 allow small error (0.5–1.5 recommended)
MAX_ERROR = 10       # 👈 overall difficulty (bigger = easier)
GRID_SIZE = 40

# 📐 Convert screen to coordinate
def screen_to_coord(x, y):
    origin_x = 320
    origin_y = 240

    real_x = (x - origin_x) / GRID_SIZE
    real_y = (origin_y - y) / GRID_SIZE

    # ✅ Snap to grid (optional but recommended)
    real_x = round(real_x)
    real_y = round(real_y)

    return real_x, real_y

# 📐 Draw grid
def draw_grid(frame):
    h, w, _ = frame.shape

    for x in range(0, w, GRID_SIZE):
        cv2.line(frame, (x,0), (x,h), (50,50,50), 1)

    for y in range(0, h, GRID_SIZE):
        cv2.line(frame, (0,y), (w,y), (50,50,50), 1)

    # axes
    cv2.line(frame, (0,240), (640,240), (255,255,255), 2)
    cv2.line(frame, (320,0), (320,480), (255,255,255), 2)

    cv2.putText(frame, "X", (600,230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

    cv2.putText(frame, "Y", (330,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

# 🎯 Accuracy calc (IMPROVED)
def calculate_accuracy():
    total_error = 0

    for i in range(3):
        ux, uy = user_points[i]
        tx, ty = target_points[i]

        # ✅ Euclidean distance (better than grid diff)
        dist = math.sqrt((ux - tx)**2 + (uy - ty)**2)

        # ✅ Apply tolerance
        if dist < TOLERANCE:
            dist = 0

        total_error += dist

    # ✅ Smooth scaling (no instant zero)
    accuracy = max(0, 100 * (1 - total_error / MAX_ERROR))

    return int(accuracy)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # ✅ Hand tracking
    frame, finger, pinch = tracker.process(frame)

    draw_grid(frame)

    # 🎯 Show target points
    for i, (x,y) in enumerate(target_points):
        cv2.putText(frame, f"P{i+1}: ({x},{y})",
                    (20, 40 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

    # ✅ PINCH CLICK
    if finger and pinch:
        fx, fy = finger

        if not clicked:
            coord = screen_to_coord(fx, fy)

            if len(user_points) < 3:
                user_points.append(coord)
                print("Point added:", coord)

                click_time = time.time()
                clicked = True

    # reset click delay
    if time.time() - click_time > 0.5:
        clicked = False

    # ✅ Draw user points
    for p in user_points:
        px = int(320 + p[0]*GRID_SIZE)
        py = int(240 - p[1]*GRID_SIZE)

        cv2.circle(frame, (px,py), 8, (0,255,0), -1)

    # ✅ Draw triangle + accuracy
    if len(user_points) == 3:
        pts = []
        for p in user_points:
            px = int(320 + p[0]*GRID_SIZE)
            py = int(240 - p[1]*GRID_SIZE)
            pts.append((px,py))

        cv2.line(frame, pts[0], pts[1], (255,0,0), 2)
        cv2.line(frame, pts[1], pts[2], (255,0,0), 2)
        cv2.line(frame, pts[2], pts[0], (255,0,0), 2)

        accuracy = calculate_accuracy()

        # 🎨 Color based on accuracy
        color = (0,255,0) if accuracy >= 70 else (0,0,255)

        cv2.putText(frame, f"Accuracy: {accuracy}%",
                    (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 3)

        msg = "GOOD!" if accuracy >= 70 else "TRY AGAIN"

        cv2.putText(frame, msg,
                    (300, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 3)

        cv2.putText(frame, "Press R to Restart",
                    (150, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2)

    cv2.imshow("Coordinate Game", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        user_points = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
