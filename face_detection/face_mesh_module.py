import cv2
import mediapipe as mp
import time


class FaceMesh:
    def __init__(self, static_mode=False, max_faces=2, detection_con=0.5, tracking_con=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_mode, self.max_faces, self.detection_con, self.tracking_con)
        # Changing thickness and radius of circles of the mesh on the face
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        # Converting image into RGB kind
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_rgb)

        faces = []

        if self.results.multi_face_landmarks:
            for face_lm in self.results.multi_face_landmarks:
                if draw:
                    # Marks and mesh on the faces
                    self.mp_draw.draw_landmarks(img, face_lm, self.mp_face_mesh.FACE_CONNECTIONS, self.draw_spec, self.draw_spec)

                face = []
                for lm_id, lm in enumerate(face_lm.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])

                faces.append(face)

        return img, faces


def main():
    # Choosing a video
    cap = cv2.VideoCapture("videos/2.mp4")
    p_time = 0
    face_mesh = FaceMesh()

    while True:
        success, img = cap.read()
        img, faces = face_mesh.find_face_mesh(img)

        # if len(faces) != 0:
        #     print(len(faces))

        # Frames per second counter logic
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Frames per second counter
        cv2.putText(img, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (240, 136, 10), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
