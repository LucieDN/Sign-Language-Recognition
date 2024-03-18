from difflib import get_close_matches
import tkinter
from tkinter import *
import pandas as pd
import os
from tkVideoPlayer import TkinterVideo
import time


mainapp = tkinter.Tk()
mainapp.title("Sign language dictionnary")
mainapp.minsize(1000,600)
mainapp.resizable(width=False, height=False)

# Create the structure
row_frame = tkinter.Frame(mainapp)
row_frame.columnconfigure(0, weight=1)
row_frame.columnconfigure(1, weight=1)
row_frame.grid(row=0, column=0, sticky="nsew")

top_frame = Frame(row_frame, width=1000, height=100, bg='blue')
top_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
top_frame.grid_propagate(0)

column_frame = tkinter.Frame(row_frame)
column_frame.columnconfigure(0, weight=1)
column_frame.columnconfigure(1, weight=1)
column_frame.grid(row=1, column=0)

# Create left and right frames
left_frame = Frame(column_frame, width=300, height=500, bg='red')
left_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
left_frame.grid_propagate(0)
left_frame.pack_propagate(0)

right_frame = Frame(column_frame, width=700, height=500, bg='green')
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
    sv_closer.set("")
    matches = get_close_matches(value.upper(), list)
    if matches:
        sv_closer.set(matches[0])
    return sv_closer

sv = tkinter.StringVar()
closer= tkinter.StringVar()
list  =['OUI', 'AUSSI', 'LS']

entry = tkinter.Entry(searching_frame, textvariable=sv, validate="focusout")
entry.grid(row = 0,column = 0,padx=5,pady=5, sticky="nsew")
button = tkinter.Button(searching_frame, width=50, text = "Rechercher", command = lambda: updateEntry(entry, closer, list))
button.grid(row = 0,column = 1,padx=5,pady=5, sticky="nsew")

# Afficher le mot le plus proche
word = tkinter.Label(left_frame, background="pink", textvariable=closer)
word.grid(row=1, column=0,padx=5,pady=5, sticky="wn")


# Structure right side
parameters_frame = Frame(right_frame, width=700, height=50, bg='brown')
parameters_frame.grid(row=0, column=0)
parameters_frame.grid_propagate(0)
parameters_frame.pack_propagate(0)
list_frame =  Frame(right_frame, width=700, height=450, bg='yellow')
list_frame.grid(row=1, column=0)
list_frame.grid_propagate(0)
list_frame.pack_propagate(0)


# Find a corresponding video
def videoSigning(word):
    df_instances = pd.read_csv('D:\DataPII\instances.csv', skipfooter=(120740-5000), engine='python')
    directory = "Database/Dataset/videos/"
    videos = []
    for i in range(len(df_instances)):
        path = directory + df_instances.loc[i,"id"] + ".mp4"
        if df_instances.loc[i, "sign"]==word and os.path.exists(path):
            videos.append(path)
    return videos

def slowDownVideo():
    return

def loop(e):
    # if the video had ended then replays the video from the beginning
    time.sleep(0.5)
    videoplayer.play()
    
def updateLink(word, sv_link):
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
videoplayer.seek(1)
# videoplayer.grid(sticky="nw")

button = tkinter.Button(parameters_frame, width=10, text = "Rafraichir", command = lambda: updateLink(closer, sv_link) )
button.grid(padx=5,pady=5, sticky="nsew")

# video_link = tkinter.Label(list_frame, background="pink", textvariable=sv_link)
# video_link.grid(row=1, column=0,padx=5,pady=5, sticky="wn")

# videoplayer.bind("<<Ended>>", videoplayer.play()) # when the video ends calls the loop function

mainapp.mainloop()



# # Display the corresponding video
# def updateVideo(closer, sv_link, pi_img):
#     updateLink(closer, sv_link)
#     cap = cv2.VideoCapture(sv_link.get())
#     ret, frame = cap.read()
#     img = None
#     if ret:
#         #Convert the frame to a Tkinter-compatible image
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (400, 300))
#         img = Image.fromarray(img)
#         img = ImageTk.PhotoImage(image=img)
#     pi_img.set(img)
#     print(pi_img)

# # Create a label in the frame
# lmain = Label(right_frame)
# lmain.grid()

# # Capture from camera
# cap =  cv2.VideoCapture(sv_link.get())#sv_link.get()
# print(cap)
# # function for video streaming
# def video_stream():
#     ret, frame = cap.read()
#     if ret:
#         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#         img = Image.fromarray(cv2image)
#         imgtk = ImageTk.PhotoImage(image=img)
#         lmain.imgtk = imgtk
#         lmain.configure(image=imgtk)
#         lmain.after(1, video_stream) 

# video_stream()

# canvas = tkinter.Canvas(right_frame, width=400, height=300, )
# canvas.grid(row=1, column=0, sticky="nw")

# cap = cv2.VideoCapture(sv_link.get())
# ret, frame = cap.read()
# img = tkinter.Image(imgtype="mp4", value=frame)
# img.update()
# canvas.create_image(0, 0, anchor="nw", image=img) 

# videoplayer = TkinterVideo(master=right_frame, scaled=True)
# videoplayer.load(sv_link.get())
# videoplayer.pack(expand=True, fill="both")

# videoplayer.play() # play the video

# myFrameNumber = 0
# # get total number of frames
# totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# # check for valid frame number
# if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
#     # set frame position
#     cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break




