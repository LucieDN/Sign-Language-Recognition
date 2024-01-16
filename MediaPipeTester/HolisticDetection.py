# A faire avant dans le terminal :
# pip install mediapipe
# pip install opencv-python

# Pour lancer le programme depuis le terminal :
# python .\MediaPipeTester\HolisticDetection.py

# https://www.youtube.com/watch?v=pG4sUNDOZFg

import mediapipe as mp
import cv2
import numpy as np


# obtenir le modèle détecteur de mains
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# gérer la caméra
cap = cv2.VideoCapture(0) #Choisir la caméra


while cap.isOpened():
    ret, frame = cap.read()#ret = "return value" inutile

    #BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# recolorer l'image parce que on récupère du BGR mais MediaPipe fonctionne avec du RGB

    # Set flag
    image.flags.writeable = False

    #Detections
    results = holistic.process(image)

    # Able to draw on this image
    image.flags.writeable = True

    #RGB to BGR
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    #Afficher le résultat
    # Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('Holistic Model Detections', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
