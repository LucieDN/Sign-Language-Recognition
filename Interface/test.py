import time
import tkinter as tk
import random
import cv2
from PIL import Image, ImageTk
from ReadVideo import VideoReader

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")
        
        cap1 = cv2.VideoCapture("Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4")
        self.vid = VideoReader._init_(self.window, cap1, 300, 200)
        cap2 = cv2.VideoCapture("Database\Dataset\videos\CLSFBI0103A_S001_B_285772_285869.mp4")
        self.vid = VideoReader._init_(self.window, cap2, 300, 200)

        self.window.mainloop()
        
App(tk.Tk(), "Sign language dictionnary")
