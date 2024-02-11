
import matplotlib.pyplot as plt
from SVM import csvForModel, PrepareData, Learning, Test


signs = ["LS", "AUSSI", "AVANCER"]
points = [0,4,8,12,16,20]
for sign1 in signs:
    for sign2 in signs:
        csvForModel(sign1, sign2)
        X_training, Y_training = PrepareData("Training")
        X_test, Y_test = PrepareData("Test")
        model = Learning(X_training, Y_training)
        print(f"{sign1}, {sign2} : {Test(model, X_test, Y_test)}")
