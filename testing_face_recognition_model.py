import cv2
import numpy as np
import face_recognition

# Load the saved frame
saved_frame = cv2.imread('debug_frame.jpg')
print(f"Loaded frame shape: {saved_frame.shape}, dtype: {saved_frame.dtype}")
print(f"Loaded frame min value: {saved_frame.min()}")
print(f"Loaded frame max value: {saved_frame.max()}")

# Convert the loaded frame to RGB if necessary
if saved_frame.ndim == 3 and saved_frame.shape[2] == 3:
    saved_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)
elif saved_frame.ndim == 2:
    saved_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_GRAY2RGB)

# Save the manually converted frame
cv2.imwrite('debug_frame_rgb.jpg', saved_frame_rgb)
print("Saved manually converted frame as debug_frame_rgb.jpg")

# Load the manually converted frame
manual_frame = cv2.imread('debug_frame_rgb.jpg')
manual_frame = cv2.cvtColor(manual_frame, cv2.COLOR_BGR2RGB)
print(f"Manual frame shape: {manual_frame.shape}, dtype: {manual_frame.dtype}")
print(f"Manual frame min value: {manual_frame.min()}")
print(f"Manual frame max value: {manual_frame.max()}")

# Try to use the face_recognition library on the manually converted frame
try:
    face_locations = face_recognition.face_locations(manual_frame, model="cnn")
    print(f"Detected {len(face_locations)} face(s) with CNN model")
except Exception as e:
    print(f"Error in face_recognition.face_locations with CNN model: {e}")

try:
    face_locations = face_recognition.face_locations(manual_frame, model="hog")
    print(f"Detected {len(face_locations)} face(s) with HOG model")
except Exception as e:
    print(f"Error in face_recognition.face_locations with HOG model: {e}")
