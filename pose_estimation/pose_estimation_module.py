import time
import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, mode=False, upper_body=False, smooth=True, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.upper_body, self.smooth, self.detection_con, self.tracking_con)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        # Marks on the body
        if self.results.pose_landmarks:
            if draw:
                # Connections between the marks
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            # Circles on the marks
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lm_list


def main():
    # Choosing a video
    cap = cv2.VideoCapture("pose_videos/4.mp4")
    p_time = 0
    pose_detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = pose_detector.find_pose(img=img)
        lm_list = pose_detector.find_position(img=img, draw=False)

        if len(lm_list) != 0:
            # Printing chosen mark
            print(lm_list[25])
            # Tracking chosen mark - for example 25
            cv2.circle(img, (lm_list[25][1], lm_list[25][2]), 10, (0, 0, 255), cv2.FILLED)

        # Frames per second counter logic
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Frames per second counter
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (240, 136, 10), 3)

        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
