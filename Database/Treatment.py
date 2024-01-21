import pandas
import cv2
import os
import moviepy
from cv2 import (VideoCapture, imshow, waitKey, destroyAllWindows,
CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)

df = pandas.read_csv('Database\Data\instances.csv', skipfooter=120341)# On récupère que le signer 1

compteur = 0
videos = []
for i in range(len(df)):
    if df.loc[i, "sign"]=="LS":# AUSSI et LS (Langue des signes) sont utilisés 22 et 39 fois dans la base de donnée téléchargée, ils constituent une bonne base pour l'apprentissage
        compteur += 1
        videos.append(df.loc[i,"id"])

# Les vidéos ont été enregistrées à 50 frames par secondes
directory = "Database/Data/videos"
for vid in videos:
    name = vid+".mp4"
    f = os.path.join(directory, name)
    # Create video capture object
    capture = VideoCapture(f)
 
    while capture.isOpened():
        
        ret, frame = capture.read()
        if not ret:
            break 
        cv2.imshow('HandTracking', frame)
        
        while cv2.waitKey(50) and 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
