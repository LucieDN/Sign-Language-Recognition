# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

#from DataProcessing import Vectorize, videoSigning
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib


def Learning(X_training, Y_training):
    clf = svm.SVC(probability=True)
    clf.fit(X_training, Y_training)
    return clf

def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
    return model.predict(Xtests)
    # count = 0
    # for i in range (len(Xtests)):
    #     if model.predict([Xtests[i]])==[ytests[i]]:
    #         count+=1
    # return count*100/len(Xtests)




model = Learning(X_training, Y_training)

#print(Test(model, X_test, Y_test))

joblib.dump(model, "DataManipulation/final_model.pkl")
#joblib.dump(X_test, "final_model.pkl")
#joblib.dump(full_pipeline, "full_pipeline.pkl")