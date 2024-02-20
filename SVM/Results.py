
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib

model = joblib.load("DataManipulation/final_model.pkl")
X_test = joblib.load("DataManipulation/X_test.pkl")
Y_test = joblib.load("DataManipulation/Y_test.pkl")

signs = ["LS", "AUSSI", "AVANCER"]
points = [0,4,8,12,16,20]

# for sign1 in signs:
#     for sign2 in signs:
#         csvForModel(sign1, sign2)
#         X_training, Y_training = PrepareData("Training")
#         X_test, Y_test = PrepareData("Test")
#         model = Learning(X_training, Y_training)
#         print(f"{sign1}, {sign2} : {Test(model, X_test, Y_test)}")
        
        
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