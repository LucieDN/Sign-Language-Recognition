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

def csvForTraining(sign1, sign2):
    df_Sign1 = pd.read_csv(f"Database/Positions/{sign1}.csv")
    df_Sign2 = pd.read_csv(f"Database/Positions/{sign2}.csv")

    df_forTraining = pd.concat([df_Sign1, df_Sign2], keys=['0','1'])

    TComparaison = np.array(df_forTraining)
    res = []
    for i in range (len(TComparaison)//6):
        res = res +  [np.concatenate([TComparaison[point+i*6][1:] for point in range(6)])]

    #df_forTraining.to_csv(f'SVM/Training.csv', index=False)
    col = []
    for i in range(len(res[0])//6):
        col += [f'xG{i}',f'yG{i}',f'zG{i}',f'xD{i}',f'yD{i}',f'zD{i}']#,f'xD{i}',f'yD{i}',f'zD{i}'
    df_enregristrement = pd.DataFrame(res, columns=col)
    df_enregristrement.transpose()
    taille1 = df_Sign1.shape[0]//6
    taille2 = df_Sign2.shape[0]//6
    Y = [0]*taille1 + [1]*taille2
    df_enregristrement.insert(loc=0, column='Type', value=Y)
    df_enregristrement.to_csv(f'SVM/Training.csv', index=False)

    return df_enregristrement


def PrepareData():
    df_training = pd.read_csv(f"SVM/Training.csv")
    X = np.array(df_training)
    
    # On efface les vidéos qui ont trop de 0 :
    indiceVid = 0 
    for video in X:
        nb0 = np.count_nonzero(video[1:] == 0)
        nbTotal = np.count_nonzero(video[1:] != np.nan)
        if nb0>0.1*nbTotal: # On ne garde que les vidéos dont 99% portent de l'information
            X = np.delete(X, indiceVid, axis=0)
        else:
            indiceVid +=1
            
    # On transforme les NaN en 0
    for i in range(len(X)):
        X = SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=0).fit_transform(X)

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


# df_training = csvForTraining("LS", "AUSSI")
# print(df_training.shape)

Data = PrepareData()
X = [Data[video][1:] for video in range(len(Data))]
Y = [Data[video][0] for video in range(len(Data))]

model = Learning(X, Y)

#print(PrepareData2("LS", "AUSSI"))
# X = PrepareData2("LS", "AUSSI")
# res = []
# nbVideos = np.arange(1,22)
# for nbVideosTraining in nbVideos:
#     X_training = np.concatenate([X[0][:nbVideosTraining],X[1][:nbVideosTraining]])
#     Y_training = [0]*nbVideosTraining + [1]*nbVideosTraining

#     X_test =  np.concatenate([X[0][nbVideosTraining:], X[1][nbVideosTraining:]]) 
#     Y_test = [0]*(len(X[0])-nbVideosTraining) + [1]*(len(X[1])-nbVideosTraining)

#     model = Learning(X_training, Y_training)
#     #print(Test(model, X_test, Y_test))

#     res.append(Test(model, X_test, Y_test))
    
# plt.plot(nbVideos, res)
# plt.show()
