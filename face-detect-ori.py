import cv2
import sys
import time
from threading import Thread
import dlib
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from tracker import EuclideanDistTracker
import face_recognition

# Initialize the Euclidean Distance Tracker
tracker = EuclideanDistTracker(minDistance=50)

# Initialize dlib face detector and shape predictor
detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
encoder = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

# Global variables
faces = []
ids = []
prevIds = []
faceData = []
scale = 0.5
isProcessingFrame = False
frameToProcess = 0


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
        if isProcessingFrame == False:
            isProcessingFrame = True
            try:
                small_frame = cv2.resize(frameToProcess, (0, 0), fx=scale, fy=scale)
            except:
                # frame is empty
                isProcessingFrame = False
                time.sleep(0.5)
                continue
            ts = time.time()
            # detect face
            face_locations = face_recognition.face_locations(small_frame)
            # face tracking
            faces = tracker.update(face_locations)
            # extract face data (x,y,w,h,id,faceImage)
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

            # finished
            processingTime = time.time() - ts
            # print('processing done in', processingTime, 'S')
            isProcessingFrame = False

        time.sleep(0.01)


def resize_with_aspect_ratio(frame, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(frame, dim, interpolation=inter)


def capture(source='rtsp://admin:cctv1234@192.168.88.14:554/11', width=None, height=None):
    global faces
    global isProcessingFrame
    global frameToProcess
    global faceData

    vid = cv2.VideoCapture(source)

    while True:
        ret, frame = vid.read()
        if not ret:
            print(f"Cannot receive frame from camera {source}. Exiting ...")
            break

        # Resize the frame while maintaining aspect ratio
        frame = resize_with_aspect_ratio(frame, width, height)

        if isProcessingFrame == False:
            frameToProcess = frame.copy()

        # Draw face data
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


if __name__ == '__main__':
    # Start the capture thread
    c0 = Thread(target=capture, args=['rtsp://admin:cctv1234@192.168.88.2:554/11', 680, 460], daemon=True).start()
    fp0 = Thread(target=faceProcessor, args=[], daemon=True).start()

    while True:
        time.sleep(1.0)
