# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

from DataProcessing import Vectorize, videoSigning
from sklearn import svm



def PrepareData(signeT1, signeT2):
    videoT1 = videoSigning(signeT1)
    videoT2 = videoSigning(signeT2)

    points = [0,4,8,12,16,20]

    T1 = [Vectorize(video, points) for video in videoT1]
    T2 = [Vectorize(video, points) for video in videoT2]

    # On complète les vecteurs par des 0 pour qu'ils aient tous la même dimension
    tailleMax = max(max([len(T1[indice]) for indice in range(len(T1))]), max([len(T2[indice]) for indice in range(len(T2))]))
    
    for listePosition in T1:
        while len(listePosition)<tailleMax:
            listePosition.append(0)
    for listePosition in T2:
        while len(listePosition)<tailleMax:
            listePosition.append(0)
            
    return T1, T2

def Learning(T1, T2, maxEntrainement):
    
    XT1 = T1[:maxEntrainement]
    XT2 = T2[:maxEntrainement]

    X = XT1 + XT2
    y = [0]*len(XT1) + [1]*len(XT2)
    
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

def Test(model, Xtests, ytests):#Renvoie le pourcentage de réussite sur les données X étiquettée selon y
    count = 0
    for i in range (len(Xtests)):
        if model.predict([Xtests[i]])==[ytests[i]]:
            count+=1
    return count*100/len(Xtests)


T1, T2 = PrepareData("LS", "AUSSI")
model = Learning(T1, T2, 10)
tests = T1[10:] + T2[10:]
res = [0]*len(T1[10:]) + [1]*len(T2[10:])
print(Test(model, tests, res))



