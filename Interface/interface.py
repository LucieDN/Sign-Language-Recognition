import tkinter as tk

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = 600, height = 500)
        self.canvas.pack()
        
        btn = tk.Button(self.window, height=1, width=10, text="Lire", command=self.getEntry)
        btn.pack()
        
        self.myEntry = tk.Entry(self.window, width=40)
        self.myEntry.pack(pady=20)

        
        self.window.mainloop()
        
    def getEntry(self):
        res = self.myEntry.get()
        print(res)

App(tk.Tk(), "Tkinter and OpenCV")
