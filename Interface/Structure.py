import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk
import PIL
import pandas as pd
import PIL.Image, PIL.ImageTk
import cv2
import os
from ReadVideo import VideoReader


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        row_frame =tk.Frame(self.window)
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(1, weight=1)
        row_frame.grid(row=0, column=0, sticky="nsew")
        
        top_frame = Frame(row_frame, width=1100, height=150, bg='blue')
        top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        column_frame =tk.Frame(row_frame)
        column_frame.columnconfigure(0, weight=1)
        column_frame.columnconfigure(1, weight=1)
        column_frame.grid(row=1, column=0)

        # Create left and right frames
        self.search = SearchingColumn(column_frame, 0)
        self.browser = VideoBrowser(column_frame, 1)        
        
        self.window.mainloop()

class SearchingColumn():
    def __init__(self, parent_window, right):
        self.frame = Frame(parent_window, width=300, height=600, bg='red')
        self.frame.grid(row=1, column=right, padx=10, pady=5,sticky="nsew")
        
    
class VideoBrowser():
    def __init__(self, parent_window, right, video_source="Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4"):
        self.frame = Frame(parent_window, width=800, height=600, bg='green')
        self.frame.grid(row=1, column=right, padx=10, pady=5, sticky="nsew")
        
        cap = cv2.VideoCapture("Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4")
        self.vid = VideoReader._init_(self.frame, cap, 300,200)

        

App(tk.Tk(), "Sign language dictionnary")
