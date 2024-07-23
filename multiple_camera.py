import cv2
import mediapipe as mp
import time
from threading import Thread
import argparse

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Global variables to store data for each camera
camera_data = {}

def faceProcessor(camera_index):
    global camera_data

    while True:
        if not camera_data[camera_index]['isProcessingFrame']:
            camera_data[camera_index]['isProcessingFrame'] = True
            if camera_data[camera_index]['frameToProcess'] is None:
                camera_data[camera_index]['isProcessingFrame'] = False
                time.sleep(0.5)
                continue

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(camera_data[camera_index]['frameToProcess'], cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            camera_data[camera_index]['faces'] = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = camera_data[camera_index]['frameToProcess'].shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    camera_data[camera_index]['faces'].append(bbox)

            camera_data[camera_index]['isProcessingFrame'] = False
        time.sleep(0.01)

def capture(camera_index):
    global camera_data

    vid = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if not camera_data[camera_index]['isProcessingFrame']:
            camera_data[camera_index]['frameToProcess'] = frame.copy()

        # Draw face bounding boxes
        for bbox in camera_data[camera_index]['faces']:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(f'Camera {camera_index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detection with multiple cameras.')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0], help='List of camera indices to use.')
    args = parser.parse_args()

    for camera_index in args.cameras:
        camera_data[camera_index] = {
            'faces': [],
            'frameToProcess': None,
            'isProcessingFrame': False
        }
        Thread(target=capture, args=(camera_index,), daemon=True).start()
        Thread(target=faceProcessor, args=(camera_index,), daemon=True).start()

    while True:
        time.sleep(1.0)
