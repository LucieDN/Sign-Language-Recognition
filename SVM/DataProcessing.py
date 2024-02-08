import pandas as pd
import cv2
import os
import moviepy
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows,
CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)
import mediapipe as mp
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('Database\Data\instances.csv', skipfooter=120341)# On récupère que le signer 1

# Téléchargement du modèle détecteur de mains
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

# videoSigning permet de récupérer le lien des fichier mp4 correspondant au mot passé en paramètre
def videoSigning(word):
    directory = "Database/Data/videos/"
    compteur = 0
    videos = []
    for i in range(len(df)):
        if df.loc[i, "sign"]==word:
            compteur += 1
            videos.append(directory + df.loc[i,"id"] + ".mp4")
    return videos

# Vectorise permet de vectoriser les points des mains détectés pour la vidéo donnée
def Vectorize(directory, points):
    capture = VideoCapture(directory)

    #data = []
    vect = [ [] for i in range(len(points))]
    while capture.isOpened():
    
        ret, frame = capture.read()
        if not ret:
            break 
        #BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# recolorer l'image parce que on récupère du BGR mais MediaPipe fonctionne avec du RGB

        # Set flag
        image.flags.writeable = False

        #Detections
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_world_landmarks:
                rang = 0
                for point in points:
                    vect[rang].append(hand.landmark[point].x)
                    vect[rang].append(hand.landmark[point].y)
                    vect[rang].append(hand.landmark[point].z)
                    rang +=1
                # for point in points:
                #     data.append(hand.landmark[point].x)# X du point 0 (paume de la main) à chaque frame
                #     data.append(hand.landmark[point].y)
                #     data.append(hand.landmark[point].z)
        else:
            for ligne in range(len(points)):
                vect[ligne].append("NaN")
                vect[ligne].append("NaN")
                vect[ligne].append("NaN")
            #data + [0, 0, 0]
        
    return vect

# On cherche à obtenir un vecteur du genre : 
#               video 1                                               video 2 
#      Frame 1                   Frame n                   Frame 1                   Frame n
# [[X0, Y0, Z0, X1, Y1, ... , X0', Y0', Z0', ... ], [X0, Y0, Z0, X1, Y1, ... , X0', Y0', Z0', ... ], ...]
#

def CreateDataFrame(video, points):
    vect = Vectorize(video, points)# Vectorise les données du point 1 pour la première vidéo
    col = ["Points\Positions par Frame"]
    for i in range(len(vect[0])//3):
        col += [f'x{i}',f'y{i}',f'z{i}']

    df = pd.DataFrame(None, columns = col)
    for i in range(len(points)):
        df.loc[len(df.index)] = [f"P{4*i}"] + vect[i]
    
    return df

def Write(sign, points):
    videos = videoSigning(sign)
    df = []
    for video in videos:
        df.append(CreateDataFrame(video, points))
        
    res = pd.concat(df, ignore_index=True, keys=[f'V{i}' for i in range(len(videos))])

    res.to_csv(f'Database/Positions/{sign}.csv', index=False)

sign = "LS"
points = [0,4,8,12,16,20]
Write(sign, points)

