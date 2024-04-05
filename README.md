
# Reconnaissance de la langue des signes

Ce projet propose une implémentation de la reconnaissance de la langue des signes en testant un modèle SVM (Support Vector Machine) et un réseau de neurone classique. Des cas d'application ont aussi été réalisés et sont accessibles dans le dossier "Application". Cela comprend une détection en direct de la langue ainsi qu'une interface dictionnaire.

## Installations

Pour tester ce projet, vous aurez besoin d'installer la version 3.11.9 de Python. Vous pouvez le télécharger depuis les sites officiels de [Python](https://www.python.org/downloads/). Pour procéder à l'installation, suivez les instructions suivantes.

### Installation de Python 3.11.9
1. Rendez-vous sur le site de [Python](https://www.python.org/downloads/) et cherchez la version 3.11.9 puis cliquez sur Download.


<img src="./images/E1.png" width="400"/>

<br/>

2. Choississez la version correspondantes à votre système d'exploitation.

<img src="./images/E2.png" width="300"/>

<br/>

3. Cochez l'option Add python.exe to PATH puis séletionnez Install Now.

<img src="./images/E3.png" width="300"/>

<br/>

4. Fermez la fenêtre de téléchargement et retournez sur le [Notebook](./NoteBook.ipynb) du projet.

<img src="./images/E4.png" width="300"/>


### Mise à jour du kernel

1. Une fois sur le Notebook, Cliquez sur la version actuelle du kernel en haut à droite.

<img src="./images/E5.png" width="500"/>

<br/>

2. Sélectionnez "Select Another Kernel".

<img src="./images/E6.png" width="600"/>

<br/>

3. Sélectionnez "Python Environments".

<img src="./images/E7.png" width="600"/>


<br/>

4. Et choississez la version 3.11.9 qui vient d'être téléchargée.

<img src="./images/E8.png" width="600"/>


### Exécution du code

Une fois que vous avez installé Python et modifié le kernel, vous pouvez exécuter le code à partir de votre éditeur de code.
Je recommande fortement d'explorer le [Notebook](./NoteBook.ipynb) en priorité car il reprend l'ensemble des travaux qui ont été réalisés dans les dossiers [DataManipulation](./DataManipulation/),  [MediaPipe](./MediaPipe/) et [Models](./Models/).

Une fois l'**intégralité** du NoteBook éxecuté, il est possible de tester les [cas d'application](./Application/) réalisés qui comporte une [détection en direct](./Application/Direct/CaptureVideo.py) ainsi qu'une [interface dictionnaire](./Application/Interface/interface.py).


### Auteurs

Ce projet a été développé par Lucie Della-Negra dans le cadre d'un projet scolaire individuel informatique à l'ENSC (Ecole Nationale Supérieure de Cognitique).