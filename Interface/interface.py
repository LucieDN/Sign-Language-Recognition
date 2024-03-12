import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk
import PIL
import pandas as pd
import PIL.Image, PIL.ImageTk
import cv2
import os

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
        
        entry = Entry(self.frame, textvariable="Rechercher")
        entry.grid()
        
    def onChange(self):
        return
    
class VideoBrowser():
    def __init__(self, parent_window, right, video_source="Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4"):
        self.frame = Frame(parent_window, width=800, height=600, bg='green')
        self.frame.grid(row=1, column=right, padx=10, pady=5, sticky="nsew")
        
        self.video_source = video_source

        # open video source
        self.vid = MyVideoCapture(video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.frame, width = self.vid.width +600, height = self.vid.height+50)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        
        self.frame.mainloop()
        
        
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.frame.after(self.delay, self.update)
    def repeat(self, video_source):
        self.vid = MyVideoCapture(video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.frame, width = self.vid.width +600, height = self.vid.height+50)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        
    def GetListofVideos(self, word):
        df_instances = pd.read_csv('D:\DataPII\instances.csv', engine='python')
        directory = "D:\DataPII/videos/"
        videos = []
        for i in range(len(df_instances)):
            path = directory + df_instances.loc[i,"id"] + ".mp4"
            if df_instances.loc[i, "sign"]==word and os.path.exists(path):
                videos.append(path)
        return videos
            
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Release the video source when the object is destroyed
        
    # def __del__(self):
    #     if self.vid.isOpened():
    #         self.vid.release()
    #     self.frame.mainloop()
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

App(tk.Tk(), "Sign language dictionnary")
