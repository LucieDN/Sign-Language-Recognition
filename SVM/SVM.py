# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

#from DataProcessing import Vectorize, videoSigning
from sklearn import svm
import joblib

X_training = joblib.load("DataManipulation/Data/X_training.pkl")
Y_training = joblib.load("DataManipulation/Data/Y_training.pkl")
X_test = joblib.load("DataManipulation/Data/X_test.pkl")
Y_test = joblib.load("DataManipulation/Data/Y_test.pkl")

def Learning(X_training, Y_training):
    clf = svm.SVC(probability=True)
    clf.fit(X_training, Y_training)
    return clf

def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
    return model.predict_proba(Xtests)
    # count = 0
    # for i in range (len(Xtests)):
    #     if model.predict([Xtests[i]])==[ytests[i]]:
    #         count+=1
    # return count*100/len(Xtests)


model = Learning(X_training, Y_training)
print(len(X_training[0]))
#print(Test(model, X_test, Y_test))

joblib.dump(model, "SVM/SVM_model.pkl")
