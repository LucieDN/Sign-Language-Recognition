# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

#from DataProcessing import Vectorize, videoSigning
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_images
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# def PrepareData(signeT1, signeT2):
#     videoT1 = videoSigning(signeT1)
#     videoT2 = videoSigning(signeT2)

#     points = [0,4,8,12,16,20]

#     T1 = [Vectorize(video, points) for video in videoT1]
#     T2 = [Vectorize(video, points) for video in videoT2]

#     # On complète les vecteurs par des 0 pour qu'ils aient tous la même dimension
#     tailleMax = max(max([len(T1[indice]) for indice in range(len(T1))]), max([len(T2[indice]) for indice in range(len(T2))]))
    
#     for listePosition in T1:
#         while len(listePosition)<tailleMax:
#             listePosition.append(0)
#     for listePosition in T2:
#         while len(listePosition)<tailleMax:
#             listePosition.append(0)
            
#     return T1, T2

# def Learning(T1, T2, maxEntrainement):
    
#     XT1 = T1[:maxEntrainement]
#     XT2 = T2[:maxEntrainement]

#     X = XT1 + XT2
#     y = [0]*len(XT1) + [1]*len(XT2)
    
#     clf = svm.SVC()
#     clf.fit(X, y)
#     return clf

# def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
#     count = 0
#     for i in range (len(Xtests)):
#         if model.predict([Xtests[i]])==[ytests[i]]:
#             count+=1
#     return count*100/len(Xtests)


# T1, T2 = PrepareData("LS", "AUSSI")
# model = Learning(T1, T2, 10)
# tests = T1[10:] + T2[10:]
# res = [0]*len(T1[10:]) + [1]*len(T2[10:])
# print(Test(model, tests, res))


def PrepareData2(sign1, sign2):
    df_Sign1 = pd.read_csv(f"Database/Positions/{sign1}.csv").transpose()
    df_Sign2 = pd.read_csv(f"Database/Positions/{sign2}.csv").transpose()

    T1 = []
    for i in range (df_Sign1.shape[1]//6):
        T1.append(pd.concat([df_Sign1[i*6+point][1:] for point in range(5)]))
    T2 = []
    for i in range (df_Sign2.shape[1]//6):
        T2.append(pd.concat([df_Sign2[i*6+point][1:] for point in range(5)]))

    T1 = np.array(T1)
    T2 = np.array(T2)
    X = [T1, T2]
    
    # On efface les vidéos qui ont trop de 0 :
    indiceWord = 0 
    for word in X:
        indiceVid = 0
        for video in word:
            nb0 = np.count_nonzero(video == 0)
            nbTotal = np.count_nonzero(video != np.nan)
            if nb0>0.05*nbTotal:
                X[indiceWord] = np.delete(word, indiceVid, axis=0)
            else:
                indiceVid +=1
        indiceWord += 1

    
    # On transforme les NaN en 0
    for i in range(len(X)):
        X[i] = SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=0).fit_transform(X[i])

    # On complète les vecteurs par des 0 pour que X[0] et X[1] aient la même dimension
    tailleMax = max(max([len(X[0][indice]) for indice in range(len(X[0]))]), max([len(X[1][indice]) for indice in range(len(X[1]))]))
    
    for i in range(len(X)):
        t = len(X[i][0])
        if t < tailleMax:
            res = []
            for j in range(len(X[i])):
                data = X[i][j].tolist()
                while len(data) < tailleMax:
                    data.append(0)
                res.append(np.array(data))
            X[i] = res
                
    return X

def Learning(X_training, Y_training):
    

    clf = svm.SVC()
    clf.fit(X_training, Y_training)
    # clf.fit(X_training[0],Y_training[0])
    return clf

def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
    count = 0
    for i in range (len(Xtests)):
        if model.predict([Xtests[i]])==[ytests[i]]:
            count+=1
    return count*100/len(Xtests)


#print(PrepareData2("LS", "AUSSI"))
X = PrepareData2("LS", "AUSSI")
res = []
nbVideos = np.arange(1,22)
for nbVideosTraining in nbVideos:
    X_training = np.concatenate([X[0][:nbVideosTraining],X[1][:nbVideosTraining]])
    Y_training = [0]*nbVideosTraining + [1]*nbVideosTraining

    X_test =  np.concatenate([X[0][nbVideosTraining:], X[1][nbVideosTraining:]]) 
    Y_test = [0]*(len(X[0])-nbVideosTraining) + [1]*(len(X[1])-nbVideosTraining)

    model = Learning(X_training, Y_training)
    #print(Test(model, X_test, Y_test))

    res.append(Test(model, X_test, Y_test))
    
plt.plot(nbVideos, res)
plt.show()
