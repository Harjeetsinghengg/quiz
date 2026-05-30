import cv2
import time
import math
from hand_tracker import HandTracker
from questions import QuestionManager

tracker = HandTracker("hand_landmarker.task")

def reset_game():
    q = QuestionManager("questions.xlsx")
    return q, -1, 0

quiz, selected_box, select_time = reset_game()

cap = cv2.VideoCapture(0)
WAIT_TIME = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame, finger = tracker.process(frame)

    if not quiz.finished():

        row = quiz.get_current()

        # Question
        cv2.putText(frame, f"Q {quiz.index+1}/5",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,255), 2)

        cv2.putText(frame, str(row["question"]),
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,0), 2)

        boxes = []
        options = [row["opt1"], row["opt2"], row["opt3"]]

        for i in range(3):
            x1, y1 = 50, 150 + i*80
            x2, y2 = 250, y1+50
            boxes.append((x1,y1,x2,y2))

            color = (255,255,255)

            # Hover
            if finger:
                fx, fy = finger
                if x1 < fx < x2 and y1 < fy < y2:
                    color = (255,200,0)

            # Selected = BLUE
            if selected_box == i:
                color = (255,0,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

            cv2.putText(frame, str(options[i]),
                        (x1+80,y1+35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

        # Selection (can change anytime)
        if finger:
            fx, fy = finger
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                if x1 < fx < x2 and y1 < fy < y2:
                    if selected_box != i:
                        selected_box = i
                        select_time = time.time()

        # Countdown
        if selected_box != -1:
            remaining = WAIT_TIME - (time.time() - select_time)

            if remaining <= 0:
                quiz.check_answer(options[selected_box])
                selected_box = -1
                time.sleep(0.3)

            else:
                cv2.putText(frame,
                            f"Confirming {int(remaining)}",
                            (300, 450),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,255,255), 2)

    else:
        h, w, _ = frame.shape

        # ✅ WIN animation
        if quiz.correct >= 3:
            t = time.time()

            # Background pulsing
            color_val = int((1 + math.sin(t * 3)) * 127)
            frame[:] = (color_val, 50, 200)

            # Moving text
            y_move = int(250 + 30 * math.sin(t * 2))

            cv2.putText(frame, "YOU WIN!",
                        (120, y_move),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255,255,255), 4)

        # ❌ LOSE animation (rotate)
        else:
            center = (w//2, h//2)
            angle = (time.time() * 50) % 360
            M = cv2.getRotationMatrix2D(center, angle, 1)
            frame = cv2.warpAffine(frame, M, (w, h))

            cv2.putText(frame, "TRY AGAIN",
                        (120, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 4)

        # Score
        cv2.putText(frame,
                    f"Score: {quiz.correct}/5",
                    (120, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

        cv2.putText(frame, "Press R to Restart",
                    (80, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

    cv2.imshow("Math Game", frame)

    key = cv2.waitKey(1) & 0xFF

    # RESET ANYTIME
    if key == ord('r'):
        quiz.reset()
        selected_box = -1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
