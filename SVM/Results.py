
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    log_loss,
)
from sklearn.linear_model import SGDClassifier


model = joblib.load("SVM/SVM_model.pkl")
X_training = joblib.load("DataManipulation/Data/X_training.pkl")
Y_training = joblib.load("DataManipulation/Data/Y_training.pkl")

X_test = joblib.load("DataManipulation/Data/X_test.pkl")
Y_test = joblib.load("DataManipulation/Data/Y_test.pkl")
      
        
#Define fictitious ground truth and prediction results
y_pred = model.predict(X_test)

# Compute accuracy: 4/6 = 2/3
acc = np.sum(y_pred == Y_test) / len(Y_test)
print(f"{acc:.3f}")


        
# The score function computes accuracy of the SGDClassifier
train_acc = model.score(X_training, Y_training)
print(f"Training accuracy: {train_acc:.05f}")

# Using cross-validation to better evaluate accuracy, using 3 folds
cv_acc = cross_val_score(model, X_training, Y_training, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_acc}")

        
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
        #cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()