# À installer avant :
# pip install scikit-learn

# Documentation : 
# https://scikit-learn.org/stable/modules/svm.html

from DataProcessing import Vectorize, videoSigning
from sklearn import svm

videoT1 = videoSigning("LS")
videoT2 = videoSigning("AUSSI")

points = [0,4,8,12,16,20]

XT1 = [Vectorize(video, points) for video in videoT1[:38]]
XT2 = [Vectorize(video, points) for video in videoT2[:21]]
tests = [Vectorize(videoT1[38], points), Vectorize(videoT2[21], points)]
print(len(videoT1))
print(len(videoT2))

# On complète les vecteurs par des 0 pour qu'ils aient tous la même dimension
tailleMax = max(max([len(XT1[indice]) for indice in range(len(XT1))]), max([len(XT2[indice]) for indice in range(len(XT2))]), max([len(tests[indice]) for indice in range(len(tests))]))

for listePosition in XT1:
    while len(listePosition)<tailleMax:
        listePosition.append(0)
for listePosition in XT2:
    while len(listePosition)<tailleMax:
        listePosition.append(0)
for listePosition in tests:
    while len(listePosition)<tailleMax:
        listePosition.append(0)

#X = [Vectorize(videoLS[0], points), Vectorize(videoLS[1], points), Vectorize(videoLS[2], points), Vectorize(videoAUSSI[0], points), Vectorize(videoAUSSI[1], points), Vectorize(videoAUSSI[2], points)]# Vectorise les données du point 1 pour la première vidéo
X = XT1 + XT2
print(len(X))
y = [0]*len(XT1) + [1]*len(XT2)

# for x in X:
#     if 
# print(len(X[0]))
# print(len(X[1]))
# print(X[0])
# print(X[1])
# print(len(X[0]) == len(X[1]))

# Apprentissage
clf = svm.SVC()
clf.fit(X, y)


print(clf.predict(tests))



