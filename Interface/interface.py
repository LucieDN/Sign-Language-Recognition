import tkinter as tk
from tkinter import *


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        row_frame =tk.Frame(self.window)
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(1, weight=1)
        row_frame.grid(row=0, column=0, sticky=tk.E+tk.W)
        
        top_frame = Frame(row_frame, width=1100, height=150, bg='blue')
        top_frame.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E+tk.W)

        column_frame =tk.Frame(row_frame)
        column_frame.columnconfigure(0, weight=1)
        column_frame.columnconfigure(1, weight=1)
        column_frame.grid(row=1, column=0)

        # Create left and right frames
        left_frame = Frame(column_frame, width=300, height=600, bg='red')
        left_frame.grid(row=1, column=0, padx=10, pady=5,sticky=tk.E+tk.W)

        right_frame = Frame(column_frame, width=800, height=600, bg='green')
        right_frame.grid(row=1, column=1, padx=10, pady=5, sticky=tk.E+tk.W)

        # self.canvas = tk.Canvas(window, width = 600, height = 500)
        # self.canvas.pack()
        
        #btn = tk.Button(self.window, height=1, width=10, text="Rechercher", command=self.getEntry)
        # btn.pack(side="top")
        
        self.myEntry = tk.Entry(self.window, width=40)
        # self.myEntry.pack(pady=20)

        
        self.window.mainloop()
        
    def getEntry(self):
        res = self.myEntry.get()
        print(res)

App(tk.Tk(), "Tkinter and OpenCV")
