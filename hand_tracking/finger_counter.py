import cv2
import time
import os
import hand_tracking_module as htm

width_cam, height_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(3, height_cam)

files_location = "fingers_images"
my_list = os.listdir(files_location)

overlay_list = []
for image_path in my_list:
    image = cv2.imread(f"{files_location}/{image_path}")
    overlay_list.append(image)

p_time = 0

hand_detector = htm.HandDetector(detection_con=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = hand_detector.find_hands(img)
    lm_list = hand_detector.find_position(img, draw=False)
    # print(lm_list)

    if len(lm_list) != 0:
        fingers = []
        
        # Thumb
        """ "<" for the left hand and ">" for the right one. """
        if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        total_num_fingers = fingers.count(1)
        # print(total_num_fingers)

        h, w, c = overlay_list[total_num_fingers - 1].shape
        img[0:h, 0:w] = overlay_list[total_num_fingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_num_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Frames per second counter logic
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Frames per second counter
    cv2.putText(img, f"FPS: {str(int(fps))}", (550, 20), cv2.FONT_HERSHEY_PLAIN, 1, (240, 136, 10), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)