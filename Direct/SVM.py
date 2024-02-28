# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

#from DataProcessing import Vectorize, videoSigning
import statistics
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score

X_training = joblib.load("DataManipulation/Data/X_training.pkl")
Y_training = joblib.load("DataManipulation/Data/Y_training.pkl")
X_test = joblib.load("DataManipulation/Data/X_test.pkl")
Y_test = joblib.load("DataManipulation/Data/Y_test.pkl")
points = [0,4,8,12,16,20]
taille = len(X_test[0])
nbFrame = taille//6//len(points)

def Prepare(X):
    for x in X:
        for point in range(len(points)):
            subVect = x[point*nbFrame * 6 : (point+1) * nbFrame * 6]
            nonNuls = [value for value in subVect if value!=0]
            moyenne = statistics.mean(nonNuls)
            subVect = [value if value!=0 else moyenne for value in subVect ]
            x[point*nbFrame * 6 : (point+1) * nbFrame * 6] = subVect
    return 

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

Prepare(X_training)
Prepare(X_test)

model = Learning(X_training, Y_training)
#print(Test(model, X_test, Y_test))

joblib.dump(model, "Direct/SVM_model.pkl")

#Define fictitious ground truth and prediction results
y_pred = model.predict(X_test)


# The score function computes accuracy
train_acc = model.score(X_training, Y_training)
print(f"Training accuracy: {train_acc:.05f}")
test_acc = np.sum(y_pred == Y_test) / len(Y_test)
print(f"Test accuracy: {test_acc:.3f}")

# Using cross-validation to better evaluate accuracy, using 3 folds
cv_acc = cross_val_score(model, X_training, Y_training, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_acc}")


# Compute performance metrics about the multiclass SGD classifier
print(classification_report(Y_training, model.predict(X_training)))

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        Y_test,
        #display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()