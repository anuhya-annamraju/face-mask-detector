# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 13:08:04 2025

@author: aannamraju
"""

# face_mask_webcam.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# 1️⃣ Load trained model
# -------------------------------
model = load_model("face_mask_detector.h5")  # Change if using .keras

# -------------------------------
# 2️⃣ Load face detector (Haar Cascade)
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# 3️⃣ Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess face for CNN
        face_img = cv2.resize(face_img, (128, 128))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)  # Shape: (1, 128, 128, 3)
        
        # Predict mask/no mask
        prediction = model.predict(face_img)
        label = "Mask" if prediction < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
