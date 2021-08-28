import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math

# PyCaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

p_time = 0

hand_detector = htm.HandDetector(detection_con=0.7)

# PyCaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()

min_vol = vol_range[0]
max_vol = vol_range[1]

vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    img = hand_detector.find_hands(img)
    lm_list = hand_detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 30 - 240
        # Volume range -65 - 0

        vol = np.interp(length, [30, 230], [min_vol, max_vol])
        vol_bar = np.interp(length, [30, 230], [400, 150])
        vol_per = np.interp(length, [30, 230], [0, 100])

        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{str(int(vol_per))}%", (10, 450), cv2.FONT_HERSHEY_PLAIN, 1, (240, 136, 10), 3)

    # Frames per second counter logic
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Frames per second counter
    cv2.putText(img, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (240, 136, 10), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
