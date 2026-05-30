import cv2
import mediapipe as mp
import math
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        fingertip = None
        pinch = False

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # draw skeleton
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0,255,255), -1)

            # index finger
            ix = int(landmarks[8].x * w)
            iy = int(landmarks[8].y * h)
            fingertip = (ix, iy)

            cv2.circle(frame, fingertip, 10, (0,0,255), -1)

            # thumb tip
            tx = int(landmarks[4].x * w)
            ty = int(landmarks[4].y * h)

            # distance
            dist = math.hypot(ix - tx, iy - ty)

            # pinch detection
            if dist < 30:
                pinch = True
                cv2.putText(frame, "CLICK",
                            (ix, iy-20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)

        return frame, fingertip, pinch