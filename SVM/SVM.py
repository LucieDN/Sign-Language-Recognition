# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

#from DataProcessing import Vectorize, videoSigning
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# csvFor Model permet d'écrire deux fichier Test.csv et Training.csv adaptés à l'entraînement et au test du modèle pour les signes passés en paramètres
def csvForModel(sign1, sign2):
    df_Sign1 = pd.read_csv(f"Database/Positions/{sign1}.csv")
    df_Sign2 = pd.read_csv(f"Database/Positions/{sign2}.csv")

    # On concatène tout ensemble pour avoir les mêmes dimensions de vecteurs (à la fois sur les données d'entraînement et de tests)
    # res porte sur chaque ligne toute l'information concernant une vidéo
    df_All = pd.concat([df_Sign1, df_Sign2])
    TComparaison = np.array(df_All)
    res = []
    for i in range (len(TComparaison)//6):
        res = res +  [np.concatenate([TComparaison[point+i*6][1:] for point in range(6)])]

    # On revient aux dataframe pour ajouter la colonne des résultats puis on reconvertit en array
    df_All = pd.DataFrame(res)
    taille1 = df_Sign1.shape[0]//6
    taille2 = df_Sign2.shape[0]//6
    Y = [0]*taille1 + [2]*taille2
    df_All.insert(loc=0, column='Type', value=Y)
    res =  np.array(df_All)
    
    # On construit les matrices d'entraînement et de test
    tableauTraining = np.concatenate([res[: taille1//2], res[taille1 + taille2//2:]])
    df_training = pd.DataFrame(tableauTraining)
    df_training.transpose()
    df_training.to_csv(f'SVM/Training.csv', index=False)
    tableauTest = np.concatenate([res[taille1//2 : taille1 + taille2//2:]])
    df_test = pd.DataFrame(tableauTest)
    df_test.transpose()
    df_test.to_csv(f'SVM/Test.csv', index=False)

    return

# PrepareData permet le prétraitement des données avant le passage dans le modèle (NaN -> 0, élimination de certaines données...)
def PrepareData(type):
    df_data = pd.read_csv(f"SVM/{type}.csv")
    Data = np.array(df_data)
    
    # On efface les vidéos qui ont trop de 0 :
    indiceVid = 0 
    for video in Data:
        nb0 = np.count_nonzero(video[1:] == 0)
        nbTotal = np.count_nonzero(video[1:] != np.nan)
        if nb0>0.1*nbTotal: # On ne garde que les vidéos dont 99% portent de l'information
            Data = np.delete(Data, indiceVid, axis=0)
        else:
            indiceVid +=1
            
    # On transforme les NaN en 0
    for i in range(len(Data)):
        Data = SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=0).fit_transform(Data)

    X = [Data[video][1:] for video in range(len(Data))]
    Y = [Data[video][0] for video in range(len(Data))]

    return X, Y

def Learning(X_training, Y_training):
    

    clf = svm.SVC()
    clf.fit(X_training, Y_training)
    return clf

def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
    count = 0
    for i in range (len(Xtests)):
        if model.predict([Xtests[i]])==[ytests[i]]:
            count+=1
    return count*100/len(Xtests)


csvForModel("AUSSI", "LS")
X_training, Y_training = PrepareData("Training")
X_test, Y_test = PrepareData("Test")
model = Learning(X_training, Y_training)
print(Test(model, X_test, Y_test))
