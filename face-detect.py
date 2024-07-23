import cv2
import sys
import time
from threading import Thread
import dlib
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from tracker import EuclideanDistTracker
tracker = EuclideanDistTracker(minDistance=50)
# import face_recognition
# # import face_recognition_models
import face_recognition_models
print("face_recognition_models is installed and can be imported")

#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
encoder = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

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
                test_frame = cv2.resize(frameToProcess, (0, 0), fx=scale, fy=scale)
                # Convert BGR to RGB
                small_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

            except:
                # frame is empty
                isProcessingFrame = False
                time.sleep(0.5)
                continue
            ts = time.time()
            # detect face
            face_locations = face_recognition.face_locations(small_frame)
            # face encoding for recognition data
            #face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            # fetch face encoding to database to let db do the face distance calculation
            #for encoding in face_encodings:
                #print(encoding)
            #    pass
            # face tracking
            faces = tracker.update(face_locations)
            # extract face data (x,y,w,h,id,faceImage)
            ids = []
            faceData = []
            for face in faces:
                t,r,b,l,id = face
                ids.append(id)
            if (prevIds != ids) or (time.time() - tsToUpdateFaceData > 1.0):
                tsToUpdateFaceData = time.time()
                prevIds = ids
                for face in faces:
                    t,r,b,l,id = face
                    t *= int(1/scale)
                    l *= int(1/scale)
                    b *= int(1/scale)
                    r *= int(1/scale)
                    x = l
                    y = t
                    w = r - l
                    h = b - t
                    x = x - int(w/4)
                    y = y - int(h/4)
                    w = w + (2*int(w/4))
                    h = h + (2*int(h/4))
                    croppedFace = frameToProcess[y:y+h, x:x+w]
                    faceData.append((x,y,w,h,id,croppedFace))
                print(ids)
            
            # finished
            processingTime = time.time()-ts
            #print('processing done in', processingTime, 'S')
            isProcessingFrame = False

        time.sleep(0.01)

def capture(source = 0):
    global faces
    global isProcessingFrame
    global frameToProcess
    global faceData
    
    vid = cv2.VideoCapture(source)

    while True:
        ret, frame = vid.read()
        #frame = cv2.resize(frame,(640,480))
        #frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # feed frame to faceProcessor if not busy
        if isProcessingFrame == False:
            frameToProcess = frame.copy()
        # draw face data
        for face in faces:
            t,r,b,l,id = face
            t *= int(1/scale)
            l *= int(1/scale)
            b *= int(1/scale)
            r *= int(1/scale)
            x = l
            y = t
            w = r - l
            h = b - t
            cv2.putText(frame, str(id),(x,y-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)

        #for fid in faceData:
        #    x,y,w,h,id,faceImage = fid
        #    cv2.imshow('face',faceImage)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    try:
        import face_recognition

        print("face_recognition imported successfully")
    except ImportError as e:
        print("Error importing face_recognition:", e)

    c0 = Thread(target=capture, args=[0,], daemon=True).start()
    fp0 = Thread(target=faceProcessor, args=[], daemon=True).start()

    while True:
        time.sleep(1.0)