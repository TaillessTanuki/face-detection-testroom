import cv2
import mediapipe as mp
import time
from threading import Thread

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Global variables
faces = []
frameToProcess = None
isProcessingFrame = False

def faceProcessor():
    global faces
    global isProcessingFrame
    global frameToProcess

    while True:
        if isProcessingFrame == False:
            isProcessingFrame = True
            if frameToProcess is None:
                isProcessingFrame = False
                time.sleep(0.5)
                continue
            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frameToProcess, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frameToProcess.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    faces.append(bbox)

            isProcessingFrame = False
        time.sleep(0.01)

def capture(source='rtsp://admin:cctv1234@192.168.88.2:554/11'):
    global faces
    global isProcessingFrame
    global frameToProcess

    vid = cv2.VideoCapture(source)

    while True:
        ret, frame = vid.read()
        if not ret:
            print(f"Cannot receive frame from camera {source}. Exiting ...")
            break
        if isProcessingFrame == False:
            frameToProcess = frame.copy()

        # Draw face bounding boxes
        for bbox in faces:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(f'Video {source}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the capture thread with source=0 for the integrated camera
    c0 = Thread(target=capture, args=['rtsp://admin:cctv1234@192.168.88.2:554/11'], daemon=True).start()
    fp0 = Thread(target=faceProcessor, args=[], daemon=True).start()

    while True:
        time.sleep(1.0)
