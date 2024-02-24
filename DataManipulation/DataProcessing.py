# Faire le traitement des données pour préparer à l'entraînement
# Enregistre un csv training et un csv test ( + un pipeline ?)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib

signs = joblib.load("DataManipulation/Data/Signs.pkl")

# csvFor Model permet d'écrire deux fichier Test.csv et Training.csv adaptés à l'entraînement et au test du modèle pour les signes passés en paramètres
def csvForModel(signs):
    df_signs = [pd.read_csv(f"Database/Positions/{sign}.csv") for sign in signs]
    df_instances = pd.read_csv('./Database/sign_to_index.csv')
                               
    # On concatène tout ensemble pour avoir les mêmes dimensions de vecteurs (à la fois sur les données d'entraînement et de tests)
    # res porte sur chaque ligne toute l'information concernant une vidéo
    df_All = pd.concat(df_signs)
    TComparaison = np.array(df_All)
    res = []
    for i in range (len(TComparaison)//6):
        res = res +  [np.concatenate([TComparaison[point+i*6][1:] for point in range(6)])]

    # On revient aux dataframe pour ajouter la colonne des résultats puis on reconvertit en array
    df_All = pd.DataFrame(res)
    taille = [df_signs[sign].shape[0]//6 for sign in range(len(signs))]
    
    Y =[]
    for sign in range(len(signs)):
        ind = 0
        while ind < df_instances.shape[0] and df_instances.loc[ind, "sign"]!=signs[sign]:
            ind+=1
        Y += [np.double(ind)]*taille[sign]
    
    X = np.array(df_All)
    
    # On construit les matrices d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
    df_training = pd.DataFrame(x_train)
    df_training.insert(loc=0, column='Type', value=y_train)
    df_training.to_csv(f'DataManipulation/Training.csv', index=False)
    df_test = pd.DataFrame(x_test)
    df_test.insert(loc=0, column='Type', value=y_test)
    df_test.to_csv(f'DataManipulation/Test.csv', index=False)

    return 

# PrepareData permet le prétraitement des données avant le passage dans le modèle (NaN -> 0, élimination de certaines données...)
def PrepareData(type):
    df_data = pd.read_csv(f"DataManipulation/{type}.csv")
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

csvForModel(signs)
X_training, Y_training = PrepareData("Training")
X_test, Y_test = PrepareData("Test")

joblib.dump(X_training, "DataManipulation/Data/X_training.pkl")
joblib.dump(Y_training, "DataManipulation/Data/Y_training.pkl")
joblib.dump(X_test, "DataManipulation/Data/X_test.pkl")
joblib.dump(Y_test, "DataManipulation/Data/Y_test.pkl")


### Implémenter un pipeline