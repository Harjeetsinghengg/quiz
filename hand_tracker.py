import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        fingertip = None

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # draw points
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

            # skeleton
            connections = [(0,1),(1,2),(2,3),(3,4),
                           (5,6),(6,7),(7,8),
                           (9,10),(10,11),(11,12),
                           (13,14),(14,15),(15,16),
                           (17,18),(18,19),(19,20)]

            for c in connections:
                x1 = int(landmarks[c[0]].x * w)
                y1 = int(landmarks[c[0]].y * h)
                x2 = int(landmarks[c[1]].x * w)
                y2 = int(landmarks[c[1]].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # index fingertip
            lm = landmarks[8]
            fx, fy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (fx, fy), 10, (0, 0, 255), -1)

            fingertip = (fx, fy)

        return frame, fingertip
