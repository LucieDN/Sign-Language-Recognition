import time
import tkinter as tk
import random
import cv2
from PIL import Image, ImageTk
from ReadVideo import VideoReader
import os
from tkVideoPlayer import TkinterVideo
import pandas as pd
# class App:
#     def __init__(self, window, window_title):
#         self.window = window
#         self.window.title(window_title)
#         self.window.geometry("1000x600")
        
#         cap1 = cv2.VideoCapture("Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4")
#         self.vid = VideoReader._init_(self.window, cap1, 300, 200)
#         cap2 = cv2.VideoCapture("Database\Dataset\videos\CLSFBI0103A_S001_B_285772_285869.mp4")
#         self.vid = VideoReader._init_(self.window, cap2, 300, 200)

#         self.window.mainloop()
        
# App(tk.Tk(), "Sign language dictionnary")
def videoSigning(word):
    df_instances = pd.read_csv('D:\DataPII\instances.csv', skipfooter=(120740-5000), engine='python')
    directory = "Database/Dataset/videos/"
    videos = []
    for i in range(len(df_instances)):
        path = directory + df_instances.loc[i,"id"] + ".mp4"
        if df_instances.loc[i, "sign"]==word and os.path.exists(path):
            videos.append(path)
    return videos

videos = videoSigning("LS")
print(videos[0])

root = tk.Tk(width=500, height=400)
videoplayer = TkinterVideo(master=root)
videoplayer.load(videos[0])

videoplayer.grid(row=1, column=0)

videoplayer.play() # play the video
videoplayer.play() # play the video

root.mainloop()