# Ce fichier concerne le passage de la base de données du format vidéos au format vectoriel.
# Il permet en finalité d'enregistrer les données sous forme de csv pour chaque liste de vidéo représentant un même mot

import pandas as pd
import cv2
from cv2 import VideoCapture
import mediapipe as mp
import joblib
import os

# if necessary, please change the following directories to your dataset folder
directory = "Database/Dataset/videos/"
df_instances = pd.read_csv('./Database/Dataset/instances.csv', skipfooter=(120740-5000), engine='python')
df_index = pd.read_csv('./Database/sign_to_index.csv')

minimum_videos = 70
signs = [df_index.loc[i, "sign"] for i in range(3)] # On se limite au 3 premiers mots de la liste
points = [0,4,8,12,16,20]

listSignsFinal = []
if os.path.exists("./DataManipulation/Data/Signs.pkl"):
    listSignsFinal = joblib.load("./DataManipulation/Data/Signs.pkl")

# Téléchargement du modèle détecteur de mains
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8)

# videoSigning permet de récupérer le lien des fichiers mp4 correspondant au mot passé en paramètre
def videoSigning(word):
    """return a list of paths of videos signing the parameter word

    Args:
        word (string): word to search in dataset
    Returns:
        list: list of string representing video paths
    """
    videos = []
    df_instances_word = df_instances.loc[df_instances['sign'] == word]
    for title in df_instances_word['id']:
        path = directory + title + ".mp4"
        if os.path.exists(path):
            videos.append(path)
    return videos

# Vectorise permet de vectoriser les points des mains détectés pour la vidéo donnée
def Vectorize(directory, points):
    """return a matrix containing the coordinates of points (according to MediaPipe) that have been extracted from the video saved in directory path

    Args:
        directory (string): path of the video that should be convert in matrix data
        points (list of int): list of index of MediaPipe points, please see documentation <https://developers.google.com/mediapipe/solutions/vision/hand_landmarker>
        
    Returns:
        nparray: matrix corresponding to the following data,
        [[XG0, YG0, ZG0, XD0, YD0, ZD0, XG1, YG1, ... , XGn, YGn, ZGn, ... ], --> P0
        [XG0, YG0, ZG0, XD0, YD0, ZD0, XG1, YG1, ... , XGn, YGn, ZGn, ... ], --> P4
        ...
        [XG0, YG0, ZG0, XD0, YD0, ZD0, XG1, YG1, ... , XGn, YGn, ZGn, ... ], --> P20
    """
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
        rang = 0
        for point in points:
            res = results.multi_hand_world_landmarks
            if res == None:
                vect[rang].append("0")
                vect[rang].append("0")
                vect[rang].append("0")
                vect[rang].append("0")
                vect[rang].append("0")
                vect[rang].append("0")
            else :
                if len(res)>=2:#Cas où j'ai les deux mains
                    hand = results.multi_hand_world_landmarks[0]
                    vect[rang].append(hand.landmark[point].x)
                    vect[rang].append(hand.landmark[point].y)
                    vect[rang].append(hand.landmark[point].z)
                    hand = results.multi_hand_world_landmarks[1]
                    vect[rang].append(hand.landmark[point].x)
                    vect[rang].append(hand.landmark[point].y)
                    vect[rang].append(hand.landmark[point].z)
                    
                elif len(res)==1:
                    res2 = results.multi_handedness[0]
                    label = res2.classification[0].label
                    hand = results.multi_hand_world_landmarks[0]
                    if label == "Left":
                        vect[rang].append(hand.landmark[point].x)
                        vect[rang].append(hand.landmark[point].y)
                        vect[rang].append(hand.landmark[point].z)
                        vect[rang].append("0")
                        vect[rang].append("0")
                        vect[rang].append("0")
                        
                    elif label == "Right" :
                        vect[rang].append("0")
                        vect[rang].append("0")
                        vect[rang].append("0")
                        vect[rang].append(hand.landmark[point].x)
                        vect[rang].append(hand.landmark[point].y)
                        vect[rang].append(hand.landmark[point].z)
            rang +=1
        
    return vect


# Créer une Dataframe synthétisant les informations d'une vidéo sous le format ci-dessus
def CreateDataFrame(video, points):
    """Create a dataframe based on what Vectorise function returns

    Args:
        video (string): path of the video that will be convert in a dataframe
        points (list of int): list of index of MediaPipe points, please see documentation <https://developers.google.com/mediapipe/solutions/vision/hand_landmarker>

    Returns:
        dataFrame: dataframe corresponding to the matrix extracted from the video
    """
    vect = Vectorize(video, points)# Vectorise les données du point 1 pour la première vidéo
    col = ["Points\Positions par Frame"]
    
    for i in range(len(vect[0])//6):
        col += [f'xG{i}',f'yG{i}',f'zG{i}',f'xD{i}',f'yD{i}',f'zD{i}']#,f'xD{i}',f'yD{i}',f'zD{i}'
    
    df = pd.DataFrame(None, columns = col)
    for i in range(len(points)):
        df.loc[len(df.index)] = [f"P{4*i}"] + vect[i]

    return df

# Write permet d'écrire sous format csv les informations portées par les vidéos où "sign" est signé
def Write(signs, points):
    """write in a csv a dataframe for each sign in signs containing the information of videos representing it

    Args:
        signs (list of stirng): list of sign that should be convert in numeric information
        points (list of int): list of index of MediaPipe points, please see documentation <https://developers.google.com/mediapipe/solutions/vision/hand_landmarker>
        
    """
    global listSignsFinal
    for sign in signs:
        final_path = f"Database\Positions\{sign}.csv"
        if  not os.path.exists(final_path):
            videos = videoSigning(sign)  
            if len(videos)>=minimum_videos:
                listSignsFinal.append(sign)
                df = []    
                for video in videos:
                    df.append(CreateDataFrame(video, points))   
                res = pd.concat(df, ignore_index=True, keys=[f'V{i}' for i in range(len(videos))])
                res.to_csv(f'./Database/Positions/{sign}.csv', index=False)
    return

Write(signs, points)
print(listSignsFinal)
joblib.dump(listSignsFinal, "DataManipulation/Data/Signs.pkl")

