from difflib import get_close_matches
import tkinter
from tkinter import *
import pandas as pd
import os
from tkVideoPlayer import TkinterVideo
import time

# if necessary, please change the following directories to your dataset folder
directory = "./Database/Dataset/videos/"
df_instances = pd.read_csv('./Database/Dataset/instances.csv', skipfooter=(120740-5000), engine='python')
df_index = pd.read_csv('./Database/sign_to_index.csv')

path = directory + "CLSFBI0103A_S001_B_251203_251361" + ".mp4"
if os.path.exists(path):
    print("OUI MG")
    
mainapp = tkinter.Tk()
mainapp.title("Sign language dictionnary")
mainapp.minsize(1000,600)
mainapp.resizable(width=False, height=False)

# Create the structure
row_frame = tkinter.Frame(mainapp)
row_frame.columnconfigure(0, weight=1)
row_frame.columnconfigure(1, weight=1)
row_frame.grid(row=0, column=0, sticky="nsew")

top_frame = Frame(row_frame, width=1000, height=100)
top_frame.rowconfigure(0, weight=1)
top_frame.rowconfigure(1, weight=1)
top_frame.columnconfigure(0, weight=1)
top_frame.columnconfigure(1, weight=1)
top_frame.columnconfigure(2, weight=1)
top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
top_frame.grid_propagate(0)

column_frame = tkinter.Frame(row_frame)
column_frame.columnconfigure(0, weight=1)
column_frame.columnconfigure(1, weight=1)
column_frame.grid(row=1, column=0)

# Format top frame
title = tkinter.Label(top_frame, text="Dictionnaire de la langue des signes", font=("Helvetica", 14, 'bold', ))
title.grid(row=0, column=0,padx=5,pady=5, sticky="wn")

label1 = tkinter.Label(top_frame, text="Français", font=("Helvetica", 12, 'bold'))
label1.grid(row=1, column=0,padx=5,pady=5, sticky="e")

button = tkinter.Button(top_frame, width=10, text = "<---------->", command = lambda: updateEntry(entry, closer, list))
button.grid(row = 1,column = 1,padx=5,pady=5)

label2 = tkinter.Label(top_frame, text="Langue des signes belge francophone", font=("Helvetica", 12, 'bold'))
label2.grid(row=1, column=2,padx=5,pady=5, sticky="w")

# Create left and right frames
left_frame = Frame(column_frame, width=300, height=500, highlightbackground="grey", highlightthickness=3)
left_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
left_frame.grid_propagate(0)
left_frame.pack_propagate(0)

right_frame = Frame(column_frame, width=700, height=500)
right_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
right_frame.rowconfigure(1, weight=1)
right_frame.rowconfigure(1, weight=1)
right_frame.grid()
right_frame.grid_propagate(0)
right_frame.pack_propagate(0)

# Create the searching barre (left)
searching_frame = Frame(left_frame, width=300, height=50, bg='grey')
searching_frame.rowconfigure(1, weight=1)
searching_frame.rowconfigure(1, weight=1)
searching_frame.columnconfigure(1, weight=1)
searching_frame.columnconfigure(1, weight=1)
searching_frame.grid()
searching_frame.grid_propagate(0)
searching_frame.pack_propagate(0)

# renvoie la valeur de la liste la plus proche
def updateEntry(entry, sv_closer, list):
    value = entry.get()
    sv_closer.set("Pas de mot correspondant")
    matches = get_close_matches(value.upper(), list)
    if matches:
        sv_closer.set(matches[0])
    return sv_closer

sv = tkinter.StringVar()
closer= tkinter.StringVar()
closer.set("Pas de mot correspondant")

list = ['OUI', 'AUSSI', 'LS', 'AVOIR']

entry = tkinter.Entry(searching_frame, textvariable=sv, validate="focusout")
entry.grid(row = 0,column = 0,padx=5,pady=5, sticky="nsew")
button = tkinter.Button(searching_frame, width=50, text = "Rechercher", command = lambda: updateEntry(entry, closer, list))
button.grid(row = 0,column = 1,padx=5,pady=5, sticky="nsew")

# Afficher le mot le plus proche
label3 = tkinter.Label(left_frame, text="Mot le plus proche de la base de données :")
label3.grid(row=1,column=0,padx=5,pady=5, sticky="nw")

word = tkinter.Label(left_frame, textvariable=closer, font=("Helvetica", 8, 'bold'))
word.grid(row=2, column=0,padx=5,pady=5, sticky="wn")


# Structure right side
parameters_frame = Frame(right_frame, width=700, height=50, bg='grey')
parameters_frame.grid(row=0, column=0)
parameters_frame.grid_propagate(0)
parameters_frame.pack_propagate(0)
list_frame =  Frame(right_frame, width=700, height=450,highlightbackground="grey", highlightthickness=3)
list_frame.grid(row=1, column=0)
list_frame.grid_propagate(0)
list_frame.pack_propagate(0)


# videoSigning permet de récupérer le lien des fichier mp4 correspondant au mot passé en paramètre
def videoSigning(word):
    """return a list of paths of videos signing the parameter word

    Args:
        word (string): word to search in dataset
    Returns:
        list: list of string representing video paths
    """
    videos = []
    df_instances_word = df_instances.loc[df_instances['sign'] == word]
    for title in df_instances_word['id']:
        path = directory + title + ".mp4"
        if os.path.exists(path):
            videos.append(path)
    return videos


def slowDownVideo():
    return

def loop(e):
    # if the video had ended then replays the video from the beginning
    time.sleep(0.5)
    videoplayer.play()
    
def updateLink(word, sv_link):
    """update the link of the video signing the word used in "Rafraichir" button

    Args:
        word (string): word to sign
        sv_link (string): link of the video to display

    Returns:
        _type_: _description_
    """
    videos = videoSigning(word.get())
    if len(videos)>0:
        sv_link.set(videos[0])
        # time.sleep(1)
        videoplayer.load(sv_link.get())
        videoplayer.play() # play the video
        videoplayer.bind("<<Ended>>", loop)
        print(videoplayer)
    print(sv_link.get())
    return sv_link

sv_link = tkinter.StringVar()
videoplayer = TkinterVideo(master=list_frame, consistant_frame_rate=False)
videoplayer.pack(expand=True, fill='both')

button = tkinter.Button(parameters_frame, width=10, text = "Rafraichir", command = lambda: updateLink(closer, sv_link) )
button.grid(row=0, column=0, padx=5, pady=5, sticky="se")

mainapp.mainloop()