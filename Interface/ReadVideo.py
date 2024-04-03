import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time

class VideoReader():
    def _init_(root, cap, width, height):
        canvas = tk.Canvas(root, width=width, height=height)
        canvas.pack()

        #self.Update(cap, canvas)
        while True:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture("Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4")
                ret, frame = cap.read()
                #break
            # Convert the frame to a Tkinter-compatible image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, height))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            # Update the canvas with the new image
            time.sleep(0.1)
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            root.update()
    
    def Update(self, cap, canvas):
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture("Database/Dataset/videos/CLSFBI0103A_S001_B_251203_251361.mp4")
            ret, frame = cap.read()
            #break
        # Convert the frame to a Tkinter-compatible image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 300))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        # Update the canvas with the new image
        time.sleep(0.1)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.Update(canvas)
            
    def delete(root, cap):
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
