## Pour la détection des points du corps (Pose)

# Pour lancer le programme depuis le terminal :
# python .\MediaPipeTester\PoseDetection.py

#https://www.youtube.com/watch?v=06TE_U21FK4

import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils#Visualiser sur l'image
mp_pose = mp.solutions.pose#Importation du modèle
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5)



# gérer la caméra
cap = cv2.VideoCapture(0) #Choisir la caméra

while cap.isOpened():
    ret, frame = cap.read()#ret = "return value" inutile

    #BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# recolorer l'image parce que on récupère du BGR mais MediaPipe fonctionne avec du RGB

    # Set flag
    image.flags.writeable = False

    #Detections
    results = pose.process(image)


    # Able to draw on this image
    image.flags.writeable = True

    #RGB to BGR
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    #Write detections in console
    #print(results)

    #Dessiner les résultats sur l'image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)#Dessiner les points sur l'image

    cv2.imshow('MediaPipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

