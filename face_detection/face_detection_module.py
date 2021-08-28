import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, detection_con=0.5):
        self.detection_con = detection_con

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection()

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)

        bound_boxes = []

        if self.results.detections:
            for lm_id, detection in enumerate(self.results.detections):
                # Drawing marks and rectangles on the faces
                # mp_draw.draw_detection(img, detection)
                bound_box_class = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bound_box = int(bound_box_class.xmin * iw), int(bound_box_class.ymin * ih), int(bound_box_class.width * iw), int(bound_box_class.height * ih)

                bound_boxes.append([lm_id, bound_box, detection.score])

                if draw:
                    self.fancy_draw(img, bound_box)
                    cv2.putText(img, f"{str(int(detection.score[0] * 100))}%", (bound_box[0], bound_box[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (7, 184, 54), 3)

        return img, bound_boxes

    def fancy_draw(self, img, bound_box, length=30, thickness=5, rectangle_thickness=1):
        x, y, w, h = bound_box
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bound_box, (7, 184, 54), rectangle_thickness)
        # Top left
        cv2.line(img, (x, y), (x + length, y), (7, 184, 54), thickness)
        cv2.line(img, (x, y), (x, y + length), (7, 184, 54), thickness)
        # Top right
        cv2.line(img, (x1, y), (x1 - length, y), (7, 184, 54), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (7, 184, 54), thickness)
        # Bottom left
        cv2.line(img, (x, y1), (x + length, y1), (7, 184, 54), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (7, 184, 54), thickness)
        # Bottom right
        cv2.line(img, (x1, y1), (x1 - length, y1), (7, 184, 54), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (7, 184, 54), thickness)


def main():
    # Choosing a video
    cap = cv2.VideoCapture("videos/2.mp4")
    p_time = 0
    face_detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bound_boxes = face_detector.find_faces(img=img)

        # Frames per second counter logic
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Frames per second counter
        cv2.putText(img, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (240, 136, 10), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
