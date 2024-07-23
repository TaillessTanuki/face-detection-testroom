import cv2
import sys
import time
from threading import Thread, Lock
import dlib
import numpy as np
from tracker import EuclideanDistTracker
import face_recognition

tracker = EuclideanDistTracker(minDistance=50)

detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
encoder = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

faces = []
ids = []
prevIds = []
faceData = []
scale = 0.5
isProcessingFrame = False
frameToProcess = None
frameLock = Lock()


def save_frame(frame, filename):
    cv2.imwrite(filename, frame)
    print(f"Saved frame to {filename}")


def inspect_frame(frame):
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame min value: {frame.min()}")
    print(f"Frame max value: {frame.max()}")


def faceProcessor():
    global faces
    global ids
    global prevIds
    global faceData
    global isProcessingFrame
    global detector
    global predictor
    global encoder
    global frameToProcess

    tsToUpdateFaceData = time.time()

    while True:
        with frameLock:
            if frameToProcess is None or frameToProcess.size == 0:
                isProcessingFrame = False
                time.sleep(0.1)
                continue
            isProcessingFrame = True

        try:
            small_frame = cv2.resize(frameToProcess, (0, 0), fx=scale, fy=scale)
            print(f"Small frame shape: {small_frame.shape}, dtype: {small_frame.dtype}")

            # Convert BGR to RGB using cv2.cvtColor
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            rgb_small_frame = rgb_small_frame.astype(np.uint8)
            print(f"Converted frame shape: {rgb_small_frame.shape}, dtype: {rgb_small_frame.dtype}")

            if rgb_small_frame.dtype != np.uint8:
                raise ValueError("Frame must be of type uint8")

            if rgb_small_frame.shape[2] != 3:
                raise ValueError("Frame must have 3 color channels (RGB)")

            inspect_frame(rgb_small_frame)
            save_frame(rgb_small_frame, "debug_frame.jpg")

        except Exception as e:
            print(f"Error resizing or converting frame: {e}")
            isProcessingFrame = False
            time.sleep(0.5)
            continue

        ts = time.time()
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            print(f"Detected {len(face_locations)} face(s) with CNN model")
            faces = tracker.update(face_locations)
            print(f"Tracked {len(faces)} face(s)")
        except Exception as e_cnn:
            print(f"Error in face_recognition.face_locations with CNN model: {e_cnn}")
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                print(f"Detected {len(face_locations)} face(s) with HOG model")
                faces = tracker.update(face_locations)
                print(f"Tracked {len(faces)} face(s)")
            except Exception as e_hog:
                print(f"Error in face_recognition.face_locations with HOG model: {e_hog}")
                isProcessingFrame = False
                time.sleep(0.5)
                continue

        ids = []
        faceData = []
        for face in faces:
            t, r, b, l, id = face
            ids.append(id)
        if (prevIds != ids) or (time.time() - tsToUpdateFaceData > 1.0):
            tsToUpdateFaceData = time.time()
            prevIds = ids
            for face in faces:
                t, r, b, l, id = face
                t *= int(1 / scale)
                l *= int(1 / scale)
                b *= int(1 / scale)
                r *= int(1 / scale)
                x = l
                y = t
                w = r - l
                h = b - t
                x = x - int(w / 4)
                y = y - int(h / 4)
                w = w + (2 * int(w / 4))
                h = h + (2 * int(h / 4))
                croppedFace = frameToProcess[y:y + h, x:x + w]
                faceData.append((x, y, w, h, id, croppedFace))
            print(ids)

        processingTime = time.time() - ts
        print('Processing done in', processingTime, 's')
        isProcessingFrame = False

        time.sleep(0.01)


def capture(source=0):
    global faces
    global isProcessingFrame
    global frameToProcess
    global frameLock

    vid = cv2.VideoCapture(source)

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Failed to capture image")
            break

        if frame is not None and frame.size != 0:
            print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"Frame min value: {frame.min()}, max value: {frame.max()}")
            with frameLock:
                frameToProcess = frame.copy()

        for face in faces:
            t, r, b, l, id = face
            t *= int(1 / scale)
            l *= int(1 / scale)
            b *= int(1 / scale)
            r *= int(1 / scale)
            x = l
            y = t
            w = r - l
            h = b - t
            cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


if __name__ == '__main__':
    # print("Available camera indices:", list_cameras())
    # Change the source index to the correct external camera index
    external_camera_index = 1  # Replace this with the correct index from list_cameras()
    c0 = Thread(target=capture, args=[external_camera_index, ], daemon=True).start()
    fp0 = Thread(target=faceProcessor, args=[], daemon=True).start()

    while True:
        time.sleep(1.0)
