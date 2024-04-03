import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk
import pandas as pd
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
        
        # word = Label(self.browser.frame, height=1, width=10, text=self.search.sv.get())
        # word.grid(sticky="nsew")
        self.window.mainloop()

class SearchingColumn():
    def __init__(self, parent_window, right):
        self.frame = Frame(parent_window, width=300, height=600, bg='red')
        self.frame.grid(row=1, column=right, padx=10, pady=5,sticky="nsew")
        
        # btn = tk.Button(self.frame, height=1, width=10, text="Rechercher", command=self.PrintEntry)
        # btn.pack(padx=10, pady=5)
        
        # self.entry = tk.Entry(self.frame, justify='center', validate='key')
        # self.entry.bind("<Return>",self.updateEntry)
        # self.entry.pack(padx=10, pady=5)
        self.sv = StringVar()
        e = Entry(self, textvariable=self.sv, validatecommand=self.entryOnChange)#, validate="focusout"
        e.grid()
        
    def entryOnChange(self):
        print(self.sv.get())
    
    # def updateEntry(self):
    #     word = self.entry.get()
    #     print(word)
    # def getEntry(self):
    #     word = self.entry.get()
    #     return word
    
class VideoBrowser():
    def __init__(self, parent_window, right):
        self.frame = Frame(parent_window, width=800, height=600, bg='green')
        self.frame.grid(row=1, column=right, padx=10, pady=5, sticky="nsew")

    def GetWord(self, word):
        label = tk.Label(text=word)
        label.grid(padx=10, pady=5)
        
    def GetListofVideos(self, word):
        df_instances = pd.read_csv('D:\DataPII\instances.csv', engine='python')
        directory = "D:\DataPII/videos/"
        videos = []
        for i in range(len(df_instances)):
            path = directory + df_instances.loc[i,"id"] + ".mp4"
            if df_instances.loc[i, "sign"]==word and os.path.exists(path):
                videos.append(path)
        return videos

    
App(tk.Tk(), "Sign language dictionnary")
