# A faire avant dans le terminal :
# pip install mediapipe
# pip install opencv-python

# Pour lancer le programme depuis le terminal :
# python .\MediaPipeTester\HandDetection.py


## Pour la détection des points des mains (Hand Landmarker)

import mediapipe as mp
import cv2
import numpy as np


# obtenir le modèle détecteur de mains
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)


# gérer la caméra
cap = cv2.VideoCapture(0) #Choisir la caméra


while cap.isOpened():
    ret, frame = cap.read()#ret = "return value" inutile
    if ret:
        
        #BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# recolorer l'image parce que on récupère du BGR mais MediaPipe fonctionne avec du RGB

        # Set flag
        image.flags.writeable = False

        #Detections
        results = hands.process(image)

        # Able to draw on this image
        image.flags.writeable = True

        #RGB to BGR
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Afficher le résultat
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):#Pour chaque résultat
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)#Dessiner les points sur l'image


        cv2.imshow('HandTracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

cap.release()
cv2.destroyAllWindows()

# Pour accéder aux résultats (positions des marquages) on fait results.multi_hand_landmarks
# --> on obtient x, y et z !