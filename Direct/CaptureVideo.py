
import statistics
import mediapipe as mp
import cv2
import os
import time
import joblib

def clearConsole():
    command = "clear"
    if os.name in ("nt", "dos"):  # If Machine is running on Windows, use cls
        command = "cls"
    os.system(command)

def findPosition(results, points, taille):
    #Detections
    
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

def Capture(nbFrame, points):
    vect = [ 0 for i in range(len(points)*nbFrame)]

    # obtenir le modèle détecteur de mains
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    # gérer la caméra
    cap = cv2.VideoCapture(0) #Choisir la caméra

    numFrame = 0
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
                for p in range(len(points)):
                    vect[p*nbFrame] = 1
                for num, hand in enumerate(results.multi_hand_landmarks):#Pour chaque résultat
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)#Dessiner les points sur l'image

            if numFrame<nbFrame:
                numFrame += 1

            cv2.imshow('HandTracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    #print(vector)
    return results

def Prepare(x):
    Data = []
    for point in range(len(points)):
        subVect = x[point*nbFrame * 6 : (point+1) * nbFrame * 6]
        nonNuls = [value for value in subVect if value!=0]
        if nonNuls:
            moyenne = statistics.mean(nonNuls)
        else:
            moyenne = 0
        subVect = [value if value!=0 else moyenne for value in subVect ]
        Data[point*nbFrame * 6 : (point+1) * nbFrame * 6] = subVect
    return Data


model = joblib.load("Direct/SVM_model.pkl")
taille =1764
points = [0,4,8,12,16,20]
nbFrame = taille//6//len(points)
vect = [0 for i in range(len(points)*nbFrame*6)]

# # obtenir le modèle détecteur de mains
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
# gérer la caméra
cap = cv2.VideoCapture(0) #Choisir la caméra

numFrame = 0
compteur = 0
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
            res = results.multi_hand_world_landmarks
            
            if len(res)==2:
                for p in range(len(points)):
                    left = results.multi_hand_world_landmarks[0]
                    vect[6*numFrame + p*nbFrame*6] =  left.landmark[p].x
                    vect[6*numFrame + p*nbFrame*6 + 1] = left.landmark[p].y
                    vect[6*numFrame + p*nbFrame*6 + 2] = left.landmark[p].z
                    right = results.multi_hand_world_landmarks[1]
                    vect[6*numFrame + p*nbFrame*6 + 3] = right.landmark[p].x
                    vect[6*numFrame + p*nbFrame*6 + 4] = right.landmark[p].y
                    vect[6*numFrame + p*nbFrame*6 + 5] = right.landmark[p].z
            elif len(res)==1:
                res2 = results.multi_handedness[0]
                label = res2.classification[0].label
                hand = results.multi_hand_world_landmarks[0]

                if label == "Left":
                    for p in range(len(points)):
                        vect[6*numFrame + p*nbFrame*6] = hand.landmark[p].x
                        vect[6*numFrame + p*nbFrame*6 + 1] = hand.landmark[p].y
                        vect[6*numFrame + p*nbFrame*6 + 2] = hand.landmark[p].z
                else:
                    for p in range(len(points)):
                        vect[6*numFrame + p*nbFrame*6 + 3] = hand.landmark[p].x
                        vect[6*numFrame + p*nbFrame*6 + 4] = hand.landmark[p].y
                        vect[6*numFrame + p*nbFrame*6 + 5] = hand.landmark[p].z

            for num, hand in enumerate(results.multi_hand_landmarks):#Pour chaque résultat
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)#Dessiner les points sur l'image
        compteur += 1
        if compteur//nbFrame:
            Data = Prepare(vect)
            sign = model.predict([vect])
            proba = model.predict_proba([vect])
            if max(proba[0])>0.7:
                clearConsole()
                print(sign)
                print(proba[0])
        
        cv2.imshow('HandTracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
            
    if numFrame<nbFrame-1:
        numFrame +=1
    else:
        vect = vect[6:]
        for i in range(6):
            vect.append(0)        

cap.release()
cv2.destroyAllWindows()

sign = model.predict([vect])
sign2 = model.predict([vect])

proba = model.predict_proba([vect])
proba2 = model.predict_proba([vect])
print(sign)
print(sign2)

print(proba)
print(proba2)